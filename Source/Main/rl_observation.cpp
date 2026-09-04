#include "rl_observation.h"

#include "rl_action.h"
#include "rl_shm_transport.h"

#include <Main/engine.h>
#include <Main/scenegraph.h>
#include <Objects/movementobject.h>
#include <Objects/riggedobject.h>
#include <Objects/itemobject.h>
#include <Editors/entity_type.h>
#include <Graphics/animationclient.h>
#include <Graphics/skeleton.h>
#include <Physics/bulletworld.h>
#include <Physics/bulletobject.h>
#include <Scripting/angelscript/ascontext.h>
#include <Scripting/angelscript/asmodule.h>
#include <Math/vec3math.h>
#include <Math/mat4.h>
#include <Math/rng_streams.h>

#include <algorithm>
#include <cmath>

namespace RLObservation {
namespace {

const int kProprioceptionKnockedOutClasses = 3;  // awake / unconscious / dead
const int kProprioceptionStateClasses = 5;        // movement / ground / attack / hit_reaction / ragdoll
const int kWeaponTypeClasses = 5;                 // none / knife / sword / big_sword / spear -- see WeaponTypeIndex()
// self.id added in schema v3 (OGRL-20260816-014): see kEntityFixedFloats's
// attacked_by_id comment -- a reward function needs the agent's own
// MovementObject id to compare against a target's attacked_by_id, and
// nothing else in the buffer exposed it before this.
//
// active_blocking/active_block_recharge/weapon_type added in schema v5
// (OGRL-20260817-028 Sec4): a policy cannot learn to use a resource
// (the active-block parry) it cannot observe being up or available, and
// cannot distinguish a spear's reach from a knife's with only a boolean
// has_weapon.
const int kProprioceptionFixedFloats = 1 /*id*/ + 3 /*pos*/ + 3 /*vel*/ + 3 /*ang_vel*/ + 3 /*facing*/ +
                                        1 /*grounded*/ + 1 /*anim_phase*/ +
                                        4 /*temp/perm/blood/block health*/ +
                                        kProprioceptionKnockedOutClasses + kProprioceptionStateClasses +
                                        1 /*has_weapon*/ +
                                        1 /*active_blocking*/ + 1 /*active_block_recharge*/ + kWeaponTypeClasses /*weapon_type*/;
// Recent action history: move_x, move_y, jump, crouch, attack, grab (6 floats/step).
const int kActionHistoryFloatsPerStep = 6;

// temp_health/blood_health added in schema v2 (OGRL-20260816-011): a reward
// function needs a denser signal than "did this entity's knocked_out flag
// flip" to shape combat behavior before knockouts are common under a
// near-random early policy -- incremental health loss on a visible opponent
// is exactly that signal. permanent_health is deliberately NOT included:
// it only drops on serious/lethal hits (sparse, same problem as knocked_out
// alone), so it doesn't add the density this is for.
//
// attacked_by_id added in schema v3 (OGRL-20260816-014): a real, serious bug
// found after training -- the training scenario (arenas/oval_arena.xml, via
// Data/Scripts/arena_level.as) spawns ~10 characters across multiple teams
// that fight EACH OTHER, not just the RL-controlled one; the reward function
// was crediting the agent for any visible entity's knockout/damage
// regardless of who caused it, meaning a large and unknown share of every
// reward signal earned in the 2M-step run (OGRL-20260816-013) was ambient
// combat between other AI characters, not the agent's own actions. Fixed by
// exposing the same script global aschar.as already tracks for exactly this
// purpose (Data/Scripts/aschar.as:389, `int attacked_by_id`, set to the
// attacker's own GetID() inside WasHit() on every hit/grab/block) -- the
// reward function can now require entity.attacked_by_id == self.id before
// crediting damage or a knockout. Not a perfect frame-accurate causation
// trace (attacked_by_id is "who hit me most recently," not timestamped, so a
// knockout immediately following someone else's hit on the same target could
// still misattribute in rare cases) but a large, well-justified improvement
// over crediting literally any visible entity's outcome.
// is_ally added in schema v4 (OGRL-20260816-015): found while re-checking for
// the SAME class of bug as attacked_by_id, not by a separate incident --
// causation alone (attacked_by_id == self.id) isn't sufficient if this
// engine ever permits an agent to damage its own teammate (whether by design
// or by an untargeted AOE/friendly-fire edge case not otherwise audited
// here); crediting that as combat success would be exactly as wrong as the
// v3 bug, just via a different path. MovementObject::ASOnSameTeam already
// exists as a direct native method (Source/Objects/movementobject.cpp) --
// no script-global lookup needed, unlike attacked_by_id.
// entity forward/anim_phase/block_health/weapon_type added in schema v5
// (OGRL-20260817-028 Sec4): forward tells the agent whether an entity is
// facing it (backstabs, punishes, all 1vN positioning); anim_phase is what a
// parry or punish is timed against ("how far into an attack is this
// opponent"); block_health exposes a failing guard; weapon_type distinguishes
// a spear's reach from a knife's. See kSchemaVersion's comment in
// rl_observation.h for why entity.time_in_state is deliberately not added.
const int kEntityFixedFloats = 1 /*valid*/ + 1 /*entity_id*/ + 3 /*rel_pos*/ + 3 /*rel_vel*/ +
                                1 /*distance*/ + 1 /*species*/ + kProprioceptionKnockedOutClasses +
                                kProprioceptionStateClasses + 1 /*is_controlled*/ + 1 /*has_weapon*/ +
                                1 /*temp_health*/ + 1 /*blood_health*/ + 1 /*attacked_by_id*/ + 1 /*is_ally*/ +
                                2 /*fwd.x,fwd.z*/ + 1 /*anim_phase*/ + 1 /*block_health*/ + kWeaponTypeClasses /*weapon_type*/;

// Reads a script global through ASModule::GetVarPtrCache -- the same
// pointer-identity cache MovementObject::NativeNeedsAnimFrames (Stage 3a)
// uses: a fixed const char* literal is looked up by string exactly once per
// module and then resolved by pointer thereafter, so this is not a VM call
// and not a per-call string lookup once warmed up. Requires the name to be
// passed as a fixed string literal from the call site (never a
// std::string-derived pointer) -- see asmodule.h's own comment on GetVarPtrCache.
int ReadIntGlobal(MovementObject* mo, const char* name) {
    void* ptr = mo->as_context->module.GetVarPtrCache(name);
    return ptr != nullptr ? *reinterpret_cast<int*>(ptr) : 0;
}
float ReadFloatGlobal(MovementObject* mo, const char* name) {
    void* ptr = mo->as_context->module.GetVarPtrCache(name);
    return ptr != nullptr ? *reinterpret_cast<float*>(ptr) : 0.0f;
}
// AngelScript's primitive bool is a single byte (matching C++ bool's layout
// on every ABI this project targets), so this follows Read{Int,Float}Global's
// exact pattern -- same GetVarPtrCache path, zero VM calls.
bool ReadBoolGlobal(MovementObject* mo, const char* name) {
    void* ptr = mo->as_context->module.GetVarPtrCache(name);
    return ptr != nullptr && *reinterpret_cast<bool*>(ptr);
}

// Maps a weapon item id (as read from a character's weapon_slots, -1 if
// unarmed) to the 5-class one-hot index this schema uses: 0=none, 1=knife,
// 2=sword, 3=big_sword, 4=spear. Reads ItemObject::item_ref()->GetLabel(),
// the same label aschar.as itself switches on (e.g. arena_level's four
// curriculum weapon types are exactly "knife"/"sword"/"big_sword"/"spear").
// "rapier" folds into the sword bucket and "staff" into the spear bucket as
// the closest reach/handling analog; any other/unrecognized label folds to
// "none" in the one-hot -- has_weapon (elsewhere in the buffer) still
// reports presence correctly even when type is unrecognized. The curriculum
// this project trains never equips anything outside the four named types, so
// the fallback path is not expected to trigger in practice.
int WeaponTypeIndex(SceneGraph* scenegraph, int weapon_item_id) {
    if (weapon_item_id == -1 || scenegraph == nullptr) {
        return 0;
    }
    Object* obj = scenegraph->GetObjectFromID(weapon_item_id);
    if (obj == nullptr || obj->GetType() != _item_object) {
        return 0;
    }
    const std::string& label = static_cast<ItemObject*>(obj)->item_ref()->GetLabel();
    if (label == "knife") return 1;
    if (label == "sword" || label == "rapier") return 2;
    if (label == "big_sword") return 3;
    if (label == "spear" || label == "staff") return 4;
    return 0;
}

struct SelfFrame {
    vec3 position;
    vec3 forward;  // flattened (yaw-only) unit forward
    vec3 right;    // flattened (yaw-only) unit right
};

vec3 ToEgocentric(const SelfFrame& frame, const vec3& world_vec) {
    // Body-relative encoding (AGENTS.md: "movement is independent of a
    // rendering camera"): rotate around the vertical axis only, so a
    // character's sense of "in front of me" / "to my right" is invariant to
    // which way it happens to be facing in world space -- this is what lets
    // the same policy weights generalize across spawn orientations and maps,
    // not just an arbitrary axis convention.
    return vec3(dot(world_vec, frame.right), world_vec.y(), dot(world_vec, frame.forward));
}

SelfFrame MakeSelfFrame(MovementObject* mo) {
    SelfFrame frame;
    frame.position = mo->position;
    vec3 facing = mo->GetFacing();
    facing = vec3(facing.x(), 0.0f, facing.z());
    float len = std::sqrt(facing.x() * facing.x() + facing.z() * facing.z());
    frame.forward = len > 1e-5f ? facing / len : vec3(0.0f, 0.0f, 1.0f);
    frame.right = vec3(frame.forward.z(), 0.0f, -frame.forward.x());
    return frame;
}

int WriteVec3(std::vector<float>* out, int offset, const vec3& v) {
    (*out)[offset++] = v.x();
    (*out)[offset++] = v.y();
    (*out)[offset++] = v.z();
    return offset;
}

int WriteOneHot(std::vector<float>* out, int offset, int classes, int index) {
    for (int i = 0; i < classes; ++i) {
        (*out)[offset + i] = (i == index) ? 1.0f : 0.0f;
    }
    return offset + classes;
}

// LOS rule v1 (kLosRuleVersion): single ray from an approximate eye point to
// the target's position, testing against ALL collision objects (not just
// static -- another character standing between observer and target genuinely
// blocks the view, which is the physically correct behavior for combat
// awareness, not an approximation). Visible iff nothing closer than the
// target is hit. Deliberately simple for v1; the plan's own FOV-hull-matched
// profile (Stage 5.2) is a distinct, separately-versioned rule, not a
// refinement of this one, since vanilla AI vision is stochastic/bone-random
// and this is deterministic by design for the main track.
bool HasLineOfSight(BulletWorld* bullet_world, const vec3& from, const vec3& to) {
    if (bullet_world == nullptr) {
        return true;  // fail open rather than blind the agent on a null world
    }
    vec3 hit_point;
    const btCollisionObject* hit = bullet_world->CheckRayCollision(from, to, &hit_point, nullptr, false);
    if (hit == nullptr) {
        return true;
    }
    // Something was hit; if it's at (or beyond) the target, treat as visible --
    // CheckRayCollision reports the closest hit up to `to`, so a hit near `to`
    // itself (the target's own collision volume) is the expected, visible case.
    const float kTargetRadiusSlack = 0.6f;  // approximate character collision radius
    return length(hit_point - to) <= kTargetRadiusSlack;
}

// FOV-matched profile (Stage 5.2, kLosRuleVersion applies to this rule too --
// see rl_observation.h): approximates the vanilla AI's own perception rule
// (Data/Scripts/aschar.as::GetVisibleCharacters/VisibilityCheck) rather than
// LOS v1's omnidirectional geometric check, for training a policy against the
// same perceptual constraints a vanilla NPC has (can be flanked, can't see
// through walls or its own peripheral-vision blind spot). Two known,
// deliberate simplifications versus the script original, both because they
// depend on internal hull/foliage-collision systems not exposed to native
// code at a reasonable cost for this stage, not because they were missed:
//  1. The script's FOV region is a hull built from a 7x7 grid of
//     quaternion-slerped sample points (GetFOVMesh) -- a "spherical
//     rectangle," not a simple cone. This uses independent horizontal/
//     vertical angular bounds instead (a rectangular angular frustum), which
//     is close in practice but not pixel-identical at the hull's corners.
//  2. The cone is oriented along the body's flattened facing (the same
//     GetFacing()-derived convention already proven for movement/egocentric
//     transforms in this function), not the head bone's own rotation. A head
//     bone's local axes are rig/bind-pose-defined with no guaranteed
//     world-space meaning -- unlike the root orientation, assuming the head
//     bone's local +Z means "look forward" would be an unverified guess. The
//     head IK chain's *position* (a measured fact, not an orientation
//     assumption) is still used, as the eye-height origin for the cone and
//     the occlusion raycast. This also sidesteps the script's own -70 literal
//     passed to mat4::SetRotationX, whose current implementation takes
//     radians with no degree conversion -- reads as a pre-existing unit
//     mismatch in the original script, not an intentional angle worth
//     reproducing.
// The stochastic bone-targeted occlusion check (VisibilityCheck) IS
// replicated faithfully -- same idea (a uniformly random bone with physics,
// redrawn until one is found; ray from the observer's head to that bone),
// same RNG stream (kGameplay, matching the script rand() call this mirrors
// after Stage 2's RNG separation -- research-log OGRL-20260815's RNG work),
// simplified only by using a single ray instead of the script's >50-unit
// segmented-raycast splitting and by omitting the separate foliage/plant
// collision pass (col.GetPlantRayCollision), which BulletWorld's public
// interface doesn't expose from here.
bool InFovCone(const vec3& local_dir, float distance, const vec3& fov_params) {
    // fov_params = (horizontal_half_angle_radians, vertical_half_angle_radians,
    // max_distance) -- read directly from the script's own fov_focus/
    // fov_peripheral globals, which GetFOVMesh consumes as raw quaternion
    // angles (radians, per quaternions.h's own constructor comment), with no
    // degree conversion in the script either -- so this uses them exactly as
    // stored, matching the units the vanilla mechanic itself operates in.
    if (fov_params.z() <= 0.0f || distance > fov_params.z()) {
        return false;
    }
    const float horizontal_angle = std::atan2(local_dir.x(), local_dir.z());
    const float vertical_angle = std::atan2(local_dir.y(), local_dir.z());
    return local_dir.z() > 0.0f && std::fabs(horizontal_angle) <= fov_params.x() && std::fabs(vertical_angle) <= fov_params.y();
}

bool HasLineOfSightNpcMatched(BulletWorld* bullet_world, MovementObject* observer, MovementObject* target,
                               const vec3& head_pos, const vec3& head_right, const vec3& head_up, const vec3& head_forward) {
    const vec3 to_target = target->position - head_pos;
    const float distance = length(to_target);
    const vec3 local_dir(dot(to_target, head_right), dot(to_target, head_up), dot(to_target, head_forward));

    const vec3* fov_focus = reinterpret_cast<vec3*>(observer->as_context->module.GetVarPtrCache("fov_focus"));
    const vec3* fov_peripheral = reinterpret_cast<vec3*>(observer->as_context->module.GetVarPtrCache("fov_peripheral"));
    const bool in_cone = (fov_focus != nullptr && InFovCone(local_dir, distance, *fov_focus)) ||
                          (fov_peripheral != nullptr && InFovCone(local_dir, distance, *fov_peripheral));
    if (!in_cone) {
        return false;
    }

    RiggedObject* target_rigged = target->rigged_object();
    vec3 aim_point = target->position;
    if (target_rigged != nullptr) {
        Skeleton& skeleton = target_rigged->skeleton();
        const int num_bones = static_cast<int>(skeleton.physics_bones.size());
        if (num_bones > 0) {
            // Uniformly redraw until a physics-enabled bone is found, exactly
            // mirroring VisibilityCheck's own loop -- bounded here (unlike
            // the script's unconditional while(true)) so a skeleton with zero
            // physics bones can't spin this forever; falls back to the
            // target's root position in that edge case.
            const int kMaxAttempts = 64;
            int chosen_bone = -1;
            for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
                const int candidate = RngStreams::RangedRandomInt(RngStreams::Stream::kGameplay, 0, num_bones - 1);
                if (skeleton.physics_bones[candidate].bullet_object != nullptr) {
                    chosen_bone = candidate;
                    break;
                }
            }
            if (chosen_bone != -1) {
                aim_point = skeleton.physics_bones[chosen_bone].bullet_object->GetPosition();
            }
        }
    }

    if (bullet_world == nullptr) {
        return true;
    }
    vec3 hit_point;
    const btCollisionObject* hit = bullet_world->CheckRayCollision(head_pos, aim_point, &hit_point, nullptr, false);
    if (hit == nullptr) {
        return true;
    }
    const float kTargetRadiusSlack = 0.3f;  // aiming at a specific bone, not the character's whole volume -- tighter slack than LOS v1
    return length(hit_point - aim_point) <= kTargetRadiusSlack;
}

}  // namespace

int ComputeBufferSize(const ObservationConfig& config) {
    return kProprioceptionFixedFloats + config.action_history_steps * kActionHistoryFloatsPerStep +
           config.max_visible_entities * kEntityFixedFloats + config.local_geometry_rays;
}

int Extract(Engine* engine, MovementObject* character, const ObservationConfig& config,
            std::vector<float>* out, bool* out_truncated) {
    if (character == nullptr || out == nullptr) {
        return 0;
    }
    const int buffer_size = ComputeBufferSize(config);
    if (static_cast<int>(out->size()) < buffer_size) {
        out->resize(buffer_size);
    }
    std::fill(out->begin(), out->begin() + buffer_size, 0.0f);
    if (out_truncated != nullptr) {
        *out_truncated = false;
    }

    SceneGraph* scenegraph = engine->GetSceneGraph();
    if (scenegraph == nullptr) {
        return 0;
    }
    SelfFrame frame = MakeSelfFrame(character);

    int offset = 0;
    // --- Proprioception ---
    (*out)[offset++] = static_cast<float>(character->GetID());
    offset = WriteVec3(out, offset, character->position);
    offset = WriteVec3(out, offset, character->velocity);
    RiggedObject* self_rigged = character->rigged_object();
    vec3 self_ang_vel = self_rigged != nullptr ? self_rigged->GetAvgAngularVelocity() : vec3(0.0f);
    offset = WriteVec3(out, offset, self_ang_vel);
    offset = WriteVec3(out, offset, frame.forward);

    const bool on_ground = ReadIntGlobal(character, "on_ground") != 0;
    (*out)[offset++] = on_ground ? 1.0f : 0.0f;
    (*out)[offset++] = self_rigged != nullptr ? self_rigged->GetAnimClient().GetNormalizedAnimTime() : 0.0f;

    (*out)[offset++] = ReadFloatGlobal(character, "temp_health");
    (*out)[offset++] = ReadFloatGlobal(character, "permanent_health");
    (*out)[offset++] = ReadFloatGlobal(character, "blood_health");
    (*out)[offset++] = ReadFloatGlobal(character, "block_health");

    offset = WriteOneHot(out, offset, kProprioceptionKnockedOutClasses,
                          std::min(std::max(ReadIntGlobal(character, "knocked_out"), 0), kProprioceptionKnockedOutClasses - 1));
    offset = WriteOneHot(out, offset, kProprioceptionStateClasses,
                          std::min(std::max(ReadIntGlobal(character, "state"), 0), kProprioceptionStateClasses - 1));

    const int primary_slot = ReadIntGlobal(character, "primary_weapon_slot");
    const int primary_weapon_item_id = character->ASGetArrayIntVar("weapon_slots", primary_slot);
    (*out)[offset++] = (primary_weapon_item_id != -1) ? 1.0f : 0.0f;

    // --- Schema v5: active-block resource + weapon type (self) ---
    (*out)[offset++] = ReadBoolGlobal(character, "active_blocking") ? 1.0f : 0.0f;
    (*out)[offset++] = ReadFloatGlobal(character, "active_block_recharge");
    offset = WriteOneHot(out, offset, kWeaponTypeClasses, WeaponTypeIndex(scenegraph, primary_weapon_item_id));

    // Recent legal action history: pulled from RLAction, which owns the
    // actual applied action state, rather than duplicated/re-tracked here --
    // one source of truth. Only populated for the character RLAction is
    // actually driving; a character receiving real/AI input (not RL-injected)
    // has no legal action trace to report and stays zero-filled, honestly.
    if (RLAction::Enabled() && character->controller_id == RLAction::ControllerId()) {
        std::vector<float> history;
        RLAction::GetRecentHistory(config.action_history_steps, &history);
        const int history_floats = std::min(static_cast<int>(history.size()),
                                             config.action_history_steps * kActionHistoryFloatsPerStep);
        for (int i = 0; i < history_floats; ++i) {
            (*out)[offset + i] = history[i];
        }
    }
    offset += config.action_history_steps * kActionHistoryFloatsPerStep;

    // --- Visible entities ---
    struct Candidate {
        MovementObject* mo;
        float distance;
    };
    std::vector<Candidate> candidates;

    const bool npc_matched = config.fov_profile == FovProfile::kNpcMatched;
    // Cone orientation uses the body's flattened facing (frame.right/forward,
    // already computed above via GetFacing() -- the same verified convention
    // as everything else in this function), not the head IK-chain bone's own
    // rotation. A head bone's local axes are rig/bind-pose-defined and not
    // guaranteed to have any particular world-space meaning (unlike the root
    // orientation, which GetFacing() derives through a known, already-proven
    // convention) -- assuming the head bone's local +Z is "look forward"
    // would be an unverified guess, not a measured fact, so this profile
    // deliberately doesn't do that. Only the head's *position* (a fact, not
    // an orientation assumption) is used, as a reasonable eye-height origin
    // for the cone and the occlusion raycast.
    vec3 head_pos = character->position;
    const vec3 head_right = frame.right;
    const vec3 head_up(0.0f, 1.0f, 0.0f);
    const vec3 head_forward = frame.forward;
    if (npc_matched && character->rigged_object() != nullptr) {
        head_pos = character->rigged_object()->GetAvgIKChainPos("head");
    }

    for (Object* object : scenegraph->movement_objects_) {
        MovementObject* other = static_cast<MovementObject*>(object);
        if (other == character) {
            continue;
        }
        const float distance = length(other->position - character->position);
        if (distance > config.local_geometry_radius * 4.0f) {
            continue;  // cheap prefilter; AGENTS.md forbids a dynamic-entity cutoff, this is not one --
                       // it only skips candidates far beyond any plausible combat-relevant distance
                       // before the (more expensive) LOS raycast, and is well outside the 50m local-geometry radius.
        }
        const bool visible = npc_matched
                                  ? HasLineOfSightNpcMatched(scenegraph->bullet_world_, character, other, head_pos, head_right, head_up, head_forward)
                                  : HasLineOfSight(scenegraph->bullet_world_, character->position, other->position);
        if (!visible) {
            continue;
        }
        candidates.push_back({other, distance});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });
    if (out_truncated != nullptr && static_cast<int>(candidates.size()) > config.max_visible_entities) {
        *out_truncated = true;
    }

    for (int slot = 0; slot < config.max_visible_entities; ++slot) {
        const int slot_offset = offset + slot * kEntityFixedFloats;
        if (slot >= static_cast<int>(candidates.size())) {
            continue;  // left zero-filled; slot_valid stays 0
        }
        MovementObject* other = candidates[slot].mo;
        int eo = slot_offset;
        (*out)[eo++] = 1.0f;  // slot_valid
        (*out)[eo++] = static_cast<float>(other->GetID());
        eo = WriteVec3(out, eo, ToEgocentric(frame, other->position - character->position));
        eo = WriteVec3(out, eo, ToEgocentric(frame, other->velocity - character->velocity));
        (*out)[eo++] = candidates[slot].distance;
        (*out)[eo++] = static_cast<float>(ReadIntGlobal(other, "species"));
        eo = WriteOneHot(out, eo, kProprioceptionKnockedOutClasses,
                          std::min(std::max(ReadIntGlobal(other, "knocked_out"), 0), kProprioceptionKnockedOutClasses - 1));
        eo = WriteOneHot(out, eo, kProprioceptionStateClasses,
                          std::min(std::max(ReadIntGlobal(other, "state"), 0), kProprioceptionStateClasses - 1));
        // In a human duel both participants are ordinary player actors, but
        // driver identity is not part of the fairness contract. Preserve the
        // training-time schema-v5 value for every non-self controlled entity
        // while an external match controller is active; the old training
        // path remains unchanged because ControllerId() is zero there.
        const bool mask_driver_identity = RLShmTransport::ControllerId() > 0 && other != character;
        (*out)[eo++] = (other->controlled && !mask_driver_identity) ? 1.0f : 0.0f;
        const int other_primary_slot = ReadIntGlobal(other, "primary_weapon_slot");
        (*out)[eo++] = (other->ASGetArrayIntVar("weapon_slots", other_primary_slot) != -1) ? 1.0f : 0.0f;
        (*out)[eo++] = ReadFloatGlobal(other, "temp_health");
        (*out)[eo++] = ReadFloatGlobal(other, "blood_health");
        (*out)[eo++] = static_cast<float>(ReadIntGlobal(other, "attacked_by_id"));
        (*out)[eo++] = character->ASOnSameTeam(other) ? 1.0f : 0.0f;

        // --- Schema v5: forward/anim_phase/block_health/weapon_type (entity) ---
        const vec3 other_fwd_egocentric = ToEgocentric(frame, MakeSelfFrame(other).forward);
        (*out)[eo++] = other_fwd_egocentric.x();
        (*out)[eo++] = other_fwd_egocentric.z();
        RiggedObject* other_rigged = other->rigged_object();
        (*out)[eo++] = other_rigged != nullptr ? other_rigged->GetAnimClient().GetNormalizedAnimTime() : 0.0f;
        (*out)[eo++] = ReadFloatGlobal(other, "block_health");
        const int other_weapon_item_id = other->ASGetArrayIntVar("weapon_slots", other_primary_slot);
        eo = WriteOneHot(out, eo, kWeaponTypeClasses, WeaponTypeIndex(scenegraph, other_weapon_item_id));
    }
    offset += config.max_visible_entities * kEntityFixedFloats;

    // --- Bounded local geometry: horizontal ray fan, egocentric angles ---
    for (int i = 0; i < config.local_geometry_rays; ++i) {
        const float angle = (2.0f * static_cast<float>(M_PI) * i) / config.local_geometry_rays;
        const vec3 dir = frame.forward * std::cos(angle) + frame.right * std::sin(angle);
        const vec3 ray_end = character->position + dir * config.local_geometry_radius;
        vec3 hit_point;
        const btCollisionObject* hit = scenegraph->bullet_world_ != nullptr
                                            ? scenegraph->bullet_world_->CheckRayCollision(character->position, ray_end, &hit_point, nullptr, true)
                                            : nullptr;
        const float distance = hit != nullptr ? length(hit_point - character->position) : config.local_geometry_radius;
        (*out)[offset + i] = distance / config.local_geometry_radius;  // normalized [0,1]
    }
    offset += config.local_geometry_rays;

    return offset;
}

bool ContainsVisibleEntity(const std::vector<float>& observation,
                           const ObservationConfig& config,
                           int entity_id,
                           int observed_floats) {
    if (entity_id < 0 || observed_floats <= 0) {
        return false;
    }
    const int entity_start = kProprioceptionFixedFloats +
                             config.action_history_steps * kActionHistoryFloatsPerStep;
    const int available = std::min(observed_floats, static_cast<int>(observation.size()));
    for (int slot = 0; slot < config.max_visible_entities; ++slot) {
        const int slot_offset = entity_start + slot * kEntityFixedFloats;
        if (slot_offset + kEntityFixedFloats > available) {
            break;
        }
        if (observation[slot_offset] > 0.5f &&
            static_cast<int>(observation[slot_offset + 1]) == entity_id) {
            return true;
        }
    }
    return false;
}

std::vector<const char*> FieldNames(const ObservationConfig& config) {
    std::vector<const char*> names = {
        "self.id",
        "self.pos.x", "self.pos.y", "self.pos.z",
        "self.vel.x", "self.vel.y", "self.vel.z",
        "self.ang_vel.x", "self.ang_vel.y", "self.ang_vel.z",
        "self.forward.x", "self.forward.y", "self.forward.z",
        "self.grounded", "self.anim_phase",
        "self.temp_health", "self.permanent_health", "self.blood_health", "self.block_health",
        "self.knocked_out.awake", "self.knocked_out.unconscious", "self.knocked_out.dead",
        "self.state.movement", "self.state.ground", "self.state.attack", "self.state.hit_reaction", "self.state.ragdoll",
        "self.has_weapon",
        "self.active_blocking", "self.active_block_recharge",
        "self.weapon_type.none", "self.weapon_type.knife", "self.weapon_type.sword", "self.weapon_type.big_sword", "self.weapon_type.spear",
    };
    for (int i = 0; i < config.action_history_steps; ++i) {
        names.push_back("self.action_history[i].move_x");
        names.push_back("self.action_history[i].move_y");
        names.push_back("self.action_history[i].jump");
        names.push_back("self.action_history[i].crouch");
        names.push_back("self.action_history[i].attack");
        names.push_back("self.action_history[i].grab");
    }
    for (int i = 0; i < config.max_visible_entities; ++i) {
        names.push_back("entity[i].valid");
        names.push_back("entity[i].id");
        names.push_back("entity[i].rel_pos.x");
        names.push_back("entity[i].rel_pos.y");
        names.push_back("entity[i].rel_pos.z");
        names.push_back("entity[i].rel_vel.x");
        names.push_back("entity[i].rel_vel.y");
        names.push_back("entity[i].rel_vel.z");
        names.push_back("entity[i].distance");
        names.push_back("entity[i].species");
        names.push_back("entity[i].knocked_out.awake");
        names.push_back("entity[i].knocked_out.unconscious");
        names.push_back("entity[i].knocked_out.dead");
        names.push_back("entity[i].state.movement");
        names.push_back("entity[i].state.ground");
        names.push_back("entity[i].state.attack");
        names.push_back("entity[i].state.hit_reaction");
        names.push_back("entity[i].state.ragdoll");
        names.push_back("entity[i].is_controlled");
        names.push_back("entity[i].has_weapon");
        names.push_back("entity[i].temp_health");
        names.push_back("entity[i].blood_health");
        names.push_back("entity[i].attacked_by_id");
        names.push_back("entity[i].is_ally");
        names.push_back("entity[i].fwd.x");
        names.push_back("entity[i].fwd.z");
        names.push_back("entity[i].anim_phase");
        names.push_back("entity[i].block_health");
        names.push_back("entity[i].weapon_type.none");
        names.push_back("entity[i].weapon_type.knife");
        names.push_back("entity[i].weapon_type.sword");
        names.push_back("entity[i].weapon_type.big_sword");
        names.push_back("entity[i].weapon_type.spear");
    }
    for (int i = 0; i < config.local_geometry_rays; ++i) {
        names.push_back("local_geometry.ray[i]");
    }
    return names;
}

}  // namespace RLObservation
