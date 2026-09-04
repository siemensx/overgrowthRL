// Stage 5.1 (research-log OGRL-20260816-007): in-engine observation extraction.
//
// Per AGENTS.md's Observation contract v0: egocentric structured state, not
// pixels. Everything here is extracted directly from native engine state
// (MovementObject/RiggedObject fields, cached AngelScript globals via
// ASModule::GetVarPtrCache -- see Source/Objects/movementobject.cpp's
// NativeNeedsAnimFrames for the same pattern) with zero VM calls in the hot
// path, and written into one pre-allocated flat float buffer per call (no
// per-step heap allocation) so this is cheap enough to run every physics step
// without becoming the bottleneck -- see Stage 5.5's cost measurement.
//
// Configurable, not hardcoded: entity cap, LOS profile, and local-geometry ray
// count are all runtime parameters (ObservationConfig), because AGENTS.md
// explicitly wants oracle-LOS, FOV-matched, and other profiles comparable
// without retraining the environment API, and the plan's own review of the
// propositions document flagged an entity cap with no truncation record as a
// contract violation -- this implementation tracks truncation explicitly.
#pragma once

#include <cstdint>
#include <vector>

class Engine;
class MovementObject;

namespace RLObservation {

// Schema version: bump whenever the buffer layout changes, so recorded
// observations and any Python-side reader can detect a mismatch instead of
// silently misinterpreting floats. Logged with every run that uses this.
constexpr int kSchemaVersion = 5;  // v3 (OGRL-20260816-014): self.id + per-entity attacked_by_id, for
                                    // reward-causation attribution. v4 (OGRL-20260816-015): per-entity
                                    // is_ally (MovementObject::ASOnSameTeam), so reward code can also
                                    // require a target be hostile, not just agent-caused -- see rl_observation.cpp
                                    // v5 (OGRL-20260817-028 Sec4): the perception fields timing-based play
                                    // is made of -- per-entity forward(2)/anim_phase(1)/block_health(1)/
                                    // weapon_type-onehot(5), self active_blocking(1)/active_block_recharge(1)/
                                    // weapon_type-onehot(5). Deliberately DROPS the plan's optional
                                    // entity.time_in_state fallback field: it was specified only as a
                                    // "cheap proxy for phase if anim_phase proves unreliable for
                                    // non-controlled characters" (Sec4), and anim_phase
                                    // (RiggedObject::GetAnimClient().GetNormalizedAnimTime()) is read the
                                    // same way for entities as for self here -- no non-controlled-character
                                    // unreliability exists to work around, so the fallback would be an
                                    // unused field. Entity block 24->33 (not 34), proprioception 28->35,
                                    // total 260->339 floats -- see rl_observation.cpp for the exact layout.

// LOS rule version, tracked separately from the schema per AGENTS.md ("use
// explicit ray/shape visibility rules and log the rule version"): version 1
// is a single ray from an approximate eye height to the target's position,
// static-geometry-only occlusion (BulletWorld::CheckRayCollision, static_col=false
// so other characters can occlude too -- see rl_observation.cpp for why).
constexpr int kLosRuleVersion = 1;

enum class FovProfile {
    kOmnidirectional,  // main-track default: 360 degrees, geometric LOS only
    kNpcMatched,        // Stage 5.2: mirrors the vanilla AI's FOV hull (Data/Scripts/aschar.as:8149)
};

struct ObservationConfig {
    int max_visible_entities = 8;     // K; AGENTS.md: cap+pad deterministically, record truncation
    int local_geometry_rays = 16;     // horizontal ray fan, evenly spaced, egocentric angles
    float local_geometry_radius = 50.0f;  // meters; AGENTS.md's provisional local-geometry radius
    int action_history_steps = 4;     // recent legal action history length
    FovProfile fov_profile = FovProfile::kOmnidirectional;
};

// Fixed per-config: proprioception block + max_visible_entities * per-entity
// block + local_geometry_rays. Call once after constructing/changing config.
int ComputeBufferSize(const ObservationConfig& config);

// Extracts one observation for `character` into `out` (must be sized
// ComputeBufferSize(config) or larger). Returns the number of floats written,
// or 0 if `character` is invalid. `out_truncated` is set if more entities
// were visible than max_visible_entities allowed (some were dropped, nearest
// max_visible_entities kept) -- always check this per AGENTS.md's contract.
int Extract(Engine* engine, MovementObject* character, const ObservationConfig& config,
            std::vector<float>* out, bool* out_truncated);

// Read-only diagnostic helper for the match overlay. It answers the same
// question as the policy's entity slots without exposing any extra state to
// the policy or changing the observation contract.
bool ContainsVisibleEntity(const std::vector<float>& observation,
                           const ObservationConfig& config,
                           int entity_id,
                           int observed_floats);

// Human-readable field-name list matching Extract()'s output order, for
// logging/debugging and for a Python-side reader to self-describe the buffer
// rather than hardcoding offsets on both sides.
std::vector<const char*> FieldNames(const ObservationConfig& config);

}  // namespace RLObservation
