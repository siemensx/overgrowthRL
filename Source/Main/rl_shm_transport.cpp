#include "rl_shm_transport.h"

#include "rl_action.h"
#include "rl_equivalence.h"

#include <Main/engine.h>
#include <Main/scenegraph.h>
#include <Game/level.h>
#include <Objects/movementobject.h>
#include <UserInput/input.h>
#include <Graphics/pxdebugdraw.h>

#include "rl_ipc_platform.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

extern bool g_debug_runtime_disable_debug_draw;

namespace RLShmTransport {
namespace {

// Fixed action-slot layout: move_x, move_y, jump, crouch, attack, grab,
// drop, walk -- the exact set RLAction exposes (rl_action.h), not just the
// subset RLObservation's action-history block tracks.
const int kActionFloats = 8;
const uint32_t kMagic = 0x4C524730;  // "0GRL" as bytes, arbitrary sanity value

#pragma pack(push, 1)
struct ShmHeader {
    uint32_t magic;
    uint32_t schema_version;
    uint32_t los_rule_version;
    uint32_t obs_floats;
    uint32_t action_floats;
    uint32_t episode_done;         // engine-written: 1 if the active episode terminal condition is met
    uint32_t shutdown_requested;   // python-written: engine exits the training run gracefully when set
    uint32_t step_counter;         // engine-written: this observation's step index, for alignment checks
    uint32_t reset_requested;      // python-written: engine reloads the training scenario when set, then clears it
    uint32_t reset_seed;           // python-written: seed for the reset just requested, consumed alongside reset_requested
    uint32_t reset_ok;             // engine-written: result of the most recently processed reset (1 success, 0 failure --
                                    // Engine::ResetRLTrainingScenario can fail, e.g. mid-campaign; python must check this
                                    // rather than assume a requested reset always succeeded)
    // OGRL-20260817-028 Sec1/Sec3.1: soft reset + per-episode curriculum
    // hook. All five below are python-written, consumed alongside
    // reset_requested (same lifecycle as reset_seed) -- python must set them
    // fresh on every reset() call, the engine does not remember a prior
    // episode's values as a default.
    uint32_t reset_mode;           // 0 = hard (ClearLoadedLevel+LoadLevel), 1 = soft (Level::Message("post_reset"), no level reload)
    float reset_difficulty;        // 0..1, forwarded to the level script as "set_rl_difficulty <f>" before post_reset
    uint32_t reset_opponents;      // 1..3 requested hostile count, forwarded as "set_rl_opponents <i>" (only 1 honored by
                                    // arena_level_1v1_unarmed.as today -- see its SetUpLevel comment on why 2/3 aren't wired yet)
    float reset_weapons;           // 0..1, probability the round is armed, forwarded as "set_rl_weapons <f>"
    uint32_t reset_species;        // 0..4, forwarded as "set_rl_species <i>" -- see arena_level_1v1_unarmed.as's CreateEnemy for the mapping
};
#pragma pack(pop)

bool g_enabled = false;
// UpdateControls (the caller of Step()) runs inside a physics catch-up loop
// that can iterate multiple times per real frame (e.g. under
// global_time_scale_mult), so Step() can be re-entered after it has already
// returned false once for this run. Without this latch, a second entry
// would post the obs semaphore and block on the action semaphore again --
// but the Python side already saw shutdown_requested and exited, so nothing
// will ever post that semaphore back, hanging the process forever. Once
// shutdown is seen, every subsequent Step() call short-circuits to false
// without touching either semaphore.
bool g_shutdown_seen = false;
int g_controller_id = -1;
RLObservation::ObservationConfig g_config;
std::string g_name;

RLIpc::ShmRegion g_shm;
ShmHeader* g_header = nullptr;
float* g_obs = nullptr;
float* g_action = nullptr;
RLIpc::SemHandle g_obs_sem = RLIpc::kInvalidSem;     // posted by engine, waited on by python
RLIpc::SemHandle g_action_sem = RLIpc::kInvalidSem;  // posted by python, waited on by engine

uint64_t g_step_counter = 0;
std::vector<float> g_scratch_obs;

// Stage 6 (OGRL-20260816-021): act_period=1 reproduces the original every-
// tick-is-a-decision behavior exactly. g_ticks_since_decision counts up to
// g_act_period-1 non-decision ticks between each real handshake; see Step().
int g_act_period = 1;
int g_ticks_since_decision = 0;
bool g_match_human_visible = false;
bool g_match_overlay_visible = true;
bool g_match_overlay_toggle_key_down = false;
bool g_match_debug_draw_saved = false;
bool g_match_debug_draw_saved_value = false;

std::string ObsSemName(const std::string& base) { return base + "o"; }
std::string ActionSemName(const std::string& base) { return base + "a"; }

// Pushes this reset's curriculum parameters into the level script BEFORE the
// reset that actually re-spawns characters (post_reset, whether reached via
// the soft path directly or re-run after a hard LoadLevel below) -- so
// SetUpLevel reads the requested values, not the previous episode's. %g
// (not %f) for the floats: fixed-notation %f is fine range-wise here (both
// are already clamped 0..1 by the level script itself on receipt), but %g
// avoids printing needless trailing zeros into the message string.
void SendScenarioMessages(Engine* engine, const ShmHeader* header) {
    SceneGraph* scenegraph = engine->GetSceneGraph();
    if (scenegraph == nullptr || scenegraph->level == nullptr) {
        return;
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "set_rl_difficulty %g", static_cast<double>(header->reset_difficulty));
    scenegraph->level->Message(buf);
    std::snprintf(buf, sizeof(buf), "set_rl_opponents %u", header->reset_opponents);
    scenegraph->level->Message(buf);
    std::snprintf(buf, sizeof(buf), "set_rl_weapons %g", static_cast<double>(header->reset_weapons));
    scenegraph->level->Message(buf);
    std::snprintf(buf, sizeof(buf), "set_rl_species %u", header->reset_species);
    scenegraph->level->Message(buf);
}

MovementObject* FindControllerCharacter(SceneGraph* scenegraph, int controller_id) {
    if (scenegraph == nullptr) {
        return nullptr;
    }
    for (Object* object : scenegraph->movement_objects_) {
        MovementObject* mo = static_cast<MovementObject*>(object);
        if (mo->controller_id == controller_id) {
            return mo;
        }
    }
    return nullptr;
}

// Match-only diagnostics are emitted immediately before the render pass.
// DebugDraw's _delete_on_draw lifetime is intentional: each frame gets one
// complete set of primitives and the next frame starts cleanly.
void DrawMatchOverlayInternal(Engine* engine) {
    if (engine == nullptr || g_controller_id <= 0) {
        return;
    }
    SceneGraph* scenegraph = engine->GetSceneGraph();
    MovementObject* agent = FindControllerCharacter(scenegraph, g_controller_id);
    MovementObject* human = FindControllerCharacter(scenegraph, 0);

    auto state = [](MovementObject* mo) -> std::string {
        if (mo == nullptr) return "--";
        return mo->ASGetIntVar("knocked_out") == 0 ? "LIVE" : "OUT";
    };
    auto combat_state = [](MovementObject* mo) -> const char* {
        if (mo == nullptr) return "--";
        switch (mo->ASGetIntVar("state")) {
            case 1: return "GROUND";
            case 2: return "ATTACK";
            case 3: return "HIT";
            case 4: return "RAGDOLL";
            default: return "MOVE";
        }
    };
    auto hp = [](MovementObject* mo) -> std::string {
        if (mo == nullptr) return "--";
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.2f", static_cast<double>(mo->ASGetFloatVar("temp_health")));
        return buffer;
    };
    auto block = [](MovementObject* mo) -> std::string {
        if (mo == nullptr) return "--";
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.2f", static_cast<double>(mo->ASGetFloatVar("block_health")));
        return buffer;
    };
    char buttons[32];
    int button_pos = 0;
    const char* button_names[] = {"J", "C", "A", "G", "D", "W"};
    for (int i = 0; i < 6; ++i) {
        if (g_action != nullptr && g_action[2 + i] > 0.5f) {
            button_pos += std::snprintf(buttons + button_pos, sizeof(buttons) - button_pos,
                                        "%s%s", button_pos == 0 ? "" : " ", button_names[i]);
        }
    }
    if (button_pos == 0) {
        std::snprintf(buttons, sizeof(buttons), "-");
    }
    char action[192];
    std::snprintf(action, sizeof(action), "INPUT [%s]  STATE %s  move %.2f %.2f",
                  buttons, combat_state(agent),
                  g_action != nullptr ? static_cast<double>(g_action[0]) : 0.0,
                  g_action != nullptr ? static_cast<double>(g_action[1]) : 0.0);
    char world_action[64];
    std::snprintf(world_action, sizeof(world_action), "[%s]  %s", buttons, combat_state(agent));
    constexpr float kOverlayTextLifetimeSeconds = 1.0f;
    engine->gui.AddDebugText("ogrl_match_agent", "CHECKPOINT  " + state(agent) + "  HP " + hp(agent), kOverlayTextLifetimeSeconds);
    engine->gui.AddDebugText("ogrl_match_human", "YOU         " + state(human) + "  HP " + hp(human), kOverlayTextLifetimeSeconds);
    engine->gui.AddDebugText("ogrl_match_action", action, kOverlayTextLifetimeSeconds);
    engine->gui.AddDebugText("ogrl_match_vision",
                             g_match_human_visible ? "VISION      AGENT SEES YOU" : "VISION      AGENT OCCLUDED",
                             kOverlayTextLifetimeSeconds);
    const float target_distance = (agent != nullptr && human != nullptr)
                                      ? length(human->position - agent->position)
                                      : -1.0f;
    char target_info[160];
    std::snprintf(target_info, sizeof(target_info), "TARGET      d %.2f / range 1.50  HP %s  BLOCK %s",
                  static_cast<double>(target_distance), hp(human).c_str(), block(human).c_str());
    char world_target[96];
    std::snprintf(world_target, sizeof(world_target), "d %.2f / 1.50   HP %s   B %s",
                  static_cast<double>(target_distance), hp(human).c_str(), block(human).c_str());
    engine->gui.AddDebugText("ogrl_match_target", target_info, kOverlayTextLifetimeSeconds);
    engine->gui.AddDebugText("ogrl_match_help", "F8 overlay  |  Rematch in 5s  |  ESC quit", kOverlayTextLifetimeSeconds);

    if (agent == nullptr || human == nullptr) {
        return;
    }

    const vec4 vision_color = g_match_human_visible
                                  ? vec4(0.12f, 1.0f, 0.38f, 1.0f)
                                  : vec4(1.0f, 0.18f, 0.22f, 1.0f);
    vec3 agent_head = agent->position + vec3(0.0f, 1.35f, 0.0f);
    vec3 human_head = human->position + vec3(0.0f, 1.35f, 0.0f);
    vec3 forward = agent->GetFacing();
    forward = vec3(forward.x(), 0.0f, forward.z());
    const float forward_length = std::sqrt(forward.x() * forward.x() + forward.z() * forward.z());
    forward = forward_length > 1e-5f ? forward / forward_length : vec3(0.0f, 0.0f, 1.0f);

    DebugDraw::Instance()->AddLine(agent_head, agent_head + forward * 1.5f,
                                   vec4(0.08f, 0.75f, 1.0f, 1.0f),
                                   _delete_on_draw, _DD_XRAY);
    DebugDraw::Instance()->AddLine(agent_head, human_head, vision_color,
                                   _delete_on_draw, _DD_XRAY);
    DebugDraw::Instance()->AddWireSphere(human_head, 0.24f, vision_color,
                                         _delete_on_draw);
    DebugDraw::Instance()->AddWireSphere(agent->position, 1.5f,
                                         vec4(1.0f, 0.78f, 0.12f, 1.0f),
                                         _delete_on_draw);
    DebugDraw::Instance()->AddText(agent_head + vec3(0.0f, 0.30f, 0.0f),
                                   g_match_human_visible ? "AGENT SEES YOU" : "AGENT OCCLUDED",
                                   0.34f, _delete_on_draw, _DD_XRAY, vision_color);
    DebugDraw::Instance()->AddText(agent_head + vec3(0.0f, 0.72f, 0.0f),
                                   world_action, 0.30f, _delete_on_draw, _DD_XRAY,
                                   vec4(1.0f, 0.9f, 0.28f, 1.0f));
    DebugDraw::Instance()->AddText(agent_head + vec3(0.0f, 1.08f, 0.0f),
                                   world_target, 0.28f, _delete_on_draw, _DD_XRAY,
                                   vec4(1.0f, 0.9f, 0.28f, 1.0f));
}

void CloseAll() {
    RLIpc::CloseShm(&g_shm);
    if (g_obs_sem != RLIpc::kInvalidSem) {
        RLIpc::CloseSem(g_obs_sem);
        g_obs_sem = RLIpc::kInvalidSem;
    }
    if (g_action_sem != RLIpc::kInvalidSem) {
        RLIpc::CloseSem(g_action_sem);
        g_action_sem = RLIpc::kInvalidSem;
    }
    g_header = nullptr;
    g_obs = nullptr;
    g_action = nullptr;
}

}  // namespace

void DrawMatchOverlay(Engine* engine) {
    if (engine == nullptr || g_controller_id <= 0) {
        return;
    }

    UpdateMatchOverlayInput(engine);

    // A match window is never an editor collision-visualization window. The
    // editor can leave these direct-render debug paths enabled across a level
    // transition, and they do not go through DebugDraw's global suppression.
    g_draw_collision = false;
    SceneGraph* scenegraph = engine->GetSceneGraph();
    if (scenegraph != nullptr && scenegraph->IsCollisionNavMeshVisible()) {
        scenegraph->SetCollisionNavMeshVisible(false);
    }

    // A few legacy script paths can enqueue persistent debug elements before
    // the render phase. Dispose them here as well as suppressing the draw, so
    // F8 removes an already-visible line/sphere immediately and stays clean
    // on every subsequent frame.
    if (!g_match_overlay_visible) {
        DebugDraw::Instance()->Dispose();
    }

    // The stock combat-debug script uses the same DebugDraw queue as the RL
    // visualization. Suppressing the queue while F8 is off is what makes the
    // hotkey cover the yellow attack-range spheres too, rather than hiding
    // only the newer RL primitives.
    g_debug_runtime_disable_debug_draw = !g_match_overlay_visible;
    if (!g_match_overlay_visible) {
        return;
    }
    DrawMatchOverlayInternal(engine);
}

void UpdateMatchOverlayInput(Engine* engine) {
    if (engine == nullptr || g_controller_id <= 0) {
        return;
    }

    int num_keyboard_keys = 0;
    const Uint8* keyboard_state = SDL_GetKeyboardState(&num_keyboard_keys);
    const bool f8_down = keyboard_state != nullptr && SDL_SCANCODE_F8 < num_keyboard_keys && keyboard_state[SDL_SCANCODE_F8] != 0;
    // A short synthetic tap can begin and end between physics ticks, so the
    // SDL held-state alone is insufficient. Keyboard's pressed edge is fed
    // by the same event path as normal player hotkeys; the latch prevents a
    // single edge from toggling twice when several ticks run in one frame.
    const bool f8_pressed = Input::Instance()->getKeyboard().wasScancodePressed(SDL_SCANCODE_F8, KIMF_ANY);
    const bool f8_edge = f8_down || f8_pressed;
    if (f8_edge && !g_match_overlay_toggle_key_down) {
        g_match_overlay_visible = !g_match_overlay_visible;
        if (!g_match_overlay_visible) {
            engine->gui.debug_text.erase("ogrl_match_agent");
            engine->gui.debug_text.erase("ogrl_match_human");
            engine->gui.debug_text.erase("ogrl_match_action");
            engine->gui.debug_text.erase("ogrl_match_vision");
            engine->gui.debug_text.erase("ogrl_match_target");
            engine->gui.debug_text.erase("ogrl_match_help");
            engine->gui.debug_text.erase("a");
            engine->gui.debug_text.erase("b");
            engine->gui.debug_text.erase("c");
        }
    }
    g_match_overlay_toggle_key_down = f8_edge;
}

bool MatchOverlayVisible() {
    return g_controller_id <= 0 || g_match_overlay_visible;
}

bool Configure(const std::string& name, int controller_id, const RLObservation::ObservationConfig& config, int act_period) {
    Shutdown();  // idempotent: drop any prior segment/semaphores first

    g_name = name;
    g_controller_id = controller_id;
    g_config = config;
    g_step_counter = 0;
    g_shutdown_seen = false;
    g_act_period = act_period > 0 ? act_period : 1;
    g_ticks_since_decision = 0;
    g_match_human_visible = false;
    g_match_overlay_visible = true;
    g_match_overlay_toggle_key_down = false;
    g_match_debug_draw_saved = true;
    g_match_debug_draw_saved_value = g_debug_runtime_disable_debug_draw;

    const int obs_floats = RLObservation::ComputeBufferSize(g_config);
    const size_t map_size = sizeof(ShmHeader) + static_cast<size_t>(obs_floats) * sizeof(float) +
                            static_cast<size_t>(kActionFloats) * sizeof(float);

    if (!RLIpc::CreateShm(g_name, map_size, &g_shm)) {
        return false;
    }

    g_header = reinterpret_cast<ShmHeader*>(g_shm.addr);
    g_obs = reinterpret_cast<float*>(reinterpret_cast<char*>(g_shm.addr) + sizeof(ShmHeader));
    g_action = g_obs + obs_floats;

    std::memset(g_header, 0, sizeof(ShmHeader));
    g_header->magic = kMagic;
    g_header->schema_version = static_cast<uint32_t>(RLObservation::kSchemaVersion);
    g_header->los_rule_version = static_cast<uint32_t>(RLObservation::kLosRuleVersion);
    g_header->obs_floats = static_cast<uint32_t>(obs_floats);
    g_header->action_floats = static_cast<uint32_t>(kActionFloats);

    // Both semaphores start at 0; RLIpc clears any stale one left by a crashed
    // run so a leftover posted count can't desync a new run.
    g_obs_sem = RLIpc::CreateSem(ObsSemName(g_name));
    g_action_sem = RLIpc::CreateSem(ActionSemName(g_name));
    if (g_obs_sem == RLIpc::kInvalidSem || g_action_sem == RLIpc::kInvalidSem) {
        CloseAll();
        return false;
    }

    g_enabled = true;
    return true;
}

bool Enabled() {
    return g_enabled;
}

int ControllerId() {
    return g_controller_id;
}

bool Step(Engine* engine) {
    if (!g_enabled || g_shutdown_seen) {
        return !g_shutdown_seen;
    }

    // Stage 6 (OGRL-20260816-021 Sec 1.3(a)/2.2(a)): only every g_act_period-th
    // tick is a real decision. On the ticks in between, RLAction::Apply()
    // (called unconditionally right after this, unchanged call site in
    // engine.cpp) re-applies whatever action was staged at the last decision
    // tick -- Input::PlayerInput's held-key semantics do the rest. No shm
    // traffic on these ticks: no observation extracted, no semaphore posted,
    // nothing to wait for -- this is the entire throughput fix, not a
    // partial one, since the 674ms-dominated round trip this used to pay
    // every physics tick (120/s) now only happens every g_act_period ticks.
    ++g_ticks_since_decision;
    if (g_ticks_since_decision < g_act_period) {
        return true;
    }
    g_ticks_since_decision = 0;

    ++g_step_counter;

    SceneGraph* scenegraph = engine->GetSceneGraph();
    MovementObject* character = nullptr;
    if (scenegraph != nullptr) {
        for (Object* object : scenegraph->movement_objects_) {
            MovementObject* mo = static_cast<MovementObject*>(object);
            if (mo->controller_id == g_controller_id) {
                character = mo;
                break;
            }
        }
    }

    bool truncated = false;
    int written = 0;
    if (character != nullptr) {
        written = RLObservation::Extract(engine, character, g_config, &g_scratch_obs, &truncated);
    }
    const int obs_floats = static_cast<int>(g_header->obs_floats);
    const int copy_count = character != nullptr ? std::min(written, obs_floats) : 0;
    if (copy_count > 0) {
        std::memcpy(g_obs, g_scratch_obs.data(), static_cast<size_t>(copy_count) * sizeof(float));
    }
    if (copy_count < obs_floats) {
        std::memset(g_obs + copy_count, 0, static_cast<size_t>(obs_floats - copy_count) * sizeof(float));
    }

    if (g_controller_id > 0 && character != nullptr && scenegraph != nullptr) {
        MovementObject* human = FindControllerCharacter(scenegraph, 0);
        g_match_human_visible = human != nullptr &&
                                RLObservation::ContainsVisibleEntity(g_scratch_obs, g_config,
                                                                     human->GetID(), written);
    }

    // The rendered human duel is intentionally not terminal on an
    // unconscious/ragdolled participant.  Overgrowth represents both a
    // recoverable knockdown and a death with `knocked_out`; ending the round
    // on every non-awake value made a non-lethal hit look like a policy reset.
    // Keep the historical training semantics for controller 0, where the
    // training episode contract treats any controlled knockout as terminal.
    const bool human_duel = g_controller_id > 0;
    bool any_controlled_player_out = character == nullptr;
    if (scenegraph != nullptr) {
        any_controlled_player_out = false;
        for (Object* object : scenegraph->movement_objects_) {
            MovementObject* mo = static_cast<MovementObject*>(object);
            const int knocked_out = mo->ASGetIntVar("knocked_out");
            const bool terminal_knockout = human_duel
                                               ? knocked_out == MovementObject::_dead
                                               : knocked_out != MovementObject::_awake;
            if (mo->controlled && terminal_knockout) {
                any_controlled_player_out = true;
                break;
            }
        }
    }
    // The human-versus-checkpoint process owns the five-second rematch window;
    // the engine only reports a terminal death, not a recoverable knockdown.
    g_header->episode_done = any_controlled_player_out ? 1u : 0u;
    g_header->step_counter = static_cast<uint32_t>(g_step_counter);

    // Publish this step's observation, then block for the corresponding
    // action -- see rl_shm_transport.h for why no lock/atomic is needed on
    // the buffers themselves (the semaphore pair already establishes the
    // release/acquire ordering).
    RLIpc::PostSem(g_obs_sem);
    RLIpc::WaitSem(g_action_sem);

    if (g_header->shutdown_requested != 0) {
        g_shutdown_seen = true;
        return false;
    }

    if (g_header->reset_requested != 0) {
        // Reloads or re-populates the scenario in-process -- either path
        // rebuilds the scenegraph's characters, so every MovementObject*
        // (including the one this call already extracted an observation
        // from, above) is invalidated; nothing below this point touches it.
        // No action was meaningfully provided this call -- the caller sent a
        // reset request instead of an action -- so RLAction is deliberately
        // left untouched here; the very next Step() call runs the normal
        // extract-and-publish path against the fresh scenario and is what
        // actually returns the reset observation to Python.
        // Canonical reset boundary (OGRL-20260820-044): the reset request is
        // not an action. Clear the previous episode's held controls and
        // history before tearing down/repopulating the world. The next
        // act-period is consequently driven by an explicit zero action, so
        // S0 cannot depend on the previous episode's final button press.
        RLAction::ResetForEpisode();
        g_ticks_since_decision = 0;
        g_step_counter = 0;
        g_header->step_counter = 0;
        g_header->episode_done = 0;

        bool ok = false;
        if (g_header->reset_mode == 1) {
            // Soft reset (OGRL-20260817-028 Sec1): reseed + timer reset +
            // Level::Message("post_reset"), no ClearLoadedLevel/LoadLevel.
            // Scenario parameters go to the script first so post_reset's
            // synchronous SetUpLevel call picks up the requested
            // difficulty/opponents/weapons/species, not the previous
            // episode's.
            SendScenarioMessages(engine, g_header);
            ok = engine->SoftResetRLTrainingScenario(g_header->reset_seed);
        } else {
            // Hard reset (Stage 4, unchanged path): full ClearLoadedLevel +
            // LoadLevel. The script's own Init() runs during LoadLevel and
            // calls SetUpLevel with ITS OWN default difficulty
            // (GetRandomDifficultyNearPlayerSkill), not the RL-requested
            // one -- so once the level is back up, push the scenario
            // parameters and re-run SetUpLevel via the same post_reset path
            // the soft reset uses. This costs one extra in-process
            // SetUpLevel call (no second LoadLevel), applied consistently on
            // every hard reset (including the periodic safety-valve one),
            // so a run alternating hard/soft resets never silently reverts
            // to an un-curriculum-controlled difficulty on the hard ones.
            ok = engine->ResetRLTrainingScenario(g_header->reset_seed);
            if (ok) {
                SendScenarioMessages(engine, g_header);
                SceneGraph* scenegraph = engine->GetSceneGraph();
                if (scenegraph != nullptr && scenegraph->level != nullptr) {
                    scenegraph->level->Message("post_reset");
                }
            }
        }
        g_header->reset_requested = 0;
        g_header->reset_ok = ok ? 1u : 0u;
        if (ok) {
            RLEquivalence::OnEpisodeReset(g_header->reset_seed,
                                          static_cast<int>(g_header->reset_mode),
                                          g_header->reset_difficulty,
                                          static_cast<int>(g_header->reset_opponents),
                                          g_header->reset_weapons,
                                          static_cast<int>(g_header->reset_species));
        }
        return true;
    }

    // Stage the received action into RLAction; the caller invokes
    // RLAction::Apply() immediately after this, which is what actually
    // writes it into Input::PlayerInput.
    if (!RLAction::Enabled()) {
        RLAction::Configure(true, g_controller_id);
    }
    RLAction::SetMoveAxes(g_action[0], g_action[1]);
    RLAction::SetButton("jump", g_action[2] > 0.5f);
    RLAction::SetButton("crouch", g_action[3] > 0.5f);
    RLAction::SetButton("attack", g_action[4] > 0.5f);
    RLAction::SetButton("grab", g_action[5] > 0.5f);
    RLAction::SetButton("drop", g_action[6] > 0.5f);
    RLAction::SetButton("walk", g_action[7] > 0.5f);

    return true;
}

void Shutdown() {
    if (g_enabled && !g_name.empty()) {
        RLIpc::UnlinkShm(g_name);
        RLIpc::UnlinkSem(ObsSemName(g_name));
        RLIpc::UnlinkSem(ActionSemName(g_name));
    }
    CloseAll();
    g_enabled = false;
    if (g_match_debug_draw_saved) {
        g_debug_runtime_disable_debug_draw = g_match_debug_draw_saved_value;
        g_match_debug_draw_saved = false;
    }
    g_controller_id = -1;
    g_match_human_visible = false;
    g_match_overlay_visible = true;
    g_match_overlay_toggle_key_down = false;
    g_name.clear();
}

}  // namespace RLShmTransport
