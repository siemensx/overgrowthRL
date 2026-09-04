#include "rl_action.h"

#include <Main/engine.h>
#include <Main/scenegraph.h>
#include <Objects/movementobject.h>
#include <UserInput/input.h>
#include <Graphics/camera.h>

#include <algorithm>
#include <array>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

namespace RLAction {
namespace {

bool g_enabled = false;
bool g_native_replay_mode = false;
int g_controller_id = -1;
int g_virtual_camera_id = -1;
float g_move_x = 0.0f;
float g_move_y = 0.0f;
std::map<std::string, bool> g_buttons;
uint64_t g_step_counter = 0;
// OGRL-20260817-031: see rl_action.h's SetScriptPeriod comment. Default 1
// preserves the original every-tick behavior exactly for any caller that
// never sets it (Stage 5.3's own timing-combo scripts, which really were
// authored per-tick).
int g_script_period = 1;
// OGRL-20260817-033: see rl_action.h's SetScriptHoldSeconds comment. Ticks
// (at the engine's fixed 120Hz physics rate), not seconds -- converted once
// at set time so the per-tick ReadyToQuit() check stays integer arithmetic.
constexpr int kFixedPhysicsHz = 120;
uint64_t g_script_hold_ticks = 0;

// Ring buffer of applied action snapshots, most recent at the front.
// RLObservation's default config asks for 4 steps of history; capped well
// above any config in practice so GetRecentHistory never has to report less
// than it was asked for once warmed up.
const size_t kMaxHistory = 64;
std::deque<std::array<float, 6>> g_history;  // {move_x, move_y, jump, crouch, attack, grab}

struct ScriptEntry {
    uint64_t step;
    float move_x, move_y;
    bool jump, crouch, attack, grab, drop, walk;
};
std::vector<ScriptEntry> g_script;  // sorted by step

// RL actions are complete per-physics-tick input snapshots. Controller 0's
// native Input::ProcessController pass clears its synthetic keys before this
// function runs, so an active RL button reaches AngelScript with count=1 and
// depth_count=1 on every held tick. External controller 1 is disabled from
// native polling and receives no such clear. Resetting here makes both paths
// exactly equivalent instead of allowing controller 1 to latch released
// buttons or turn a held action into an ever-increasing depth_count.
void ApplyKey(PlayerInput& control, const std::string& name, bool active, float depth) {
    KeyState& state = control.key_down[name];
    state.count = 0;
    state.depth_count = 0;
    state.depth = 0.0f;
    if (!active) {
        return;
    }
    state.count = 1;
    state.depth_count = 1;
    state.depth = depth;
}

}  // namespace

void Configure(bool enabled, int controller_id) {
    if (g_virtual_camera_id >= 255) {
        ActiveCameras::Instance()->FreeVirtualCameraInstance(g_virtual_camera_id);
    }
    g_enabled = enabled;
    g_native_replay_mode = false;
    g_controller_id = controller_id;
    g_virtual_camera_id = -1;
    g_move_x = 0.0f;
    g_move_y = 0.0f;
    g_buttons.clear();
    g_step_counter = 0;
    g_script.clear();
    g_script_period = 1;
    g_script_hold_ticks = 0;
    g_history.clear();
}

void SetScriptPeriod(int ticks_per_decision) {
    g_script_period = std::max(1, ticks_per_decision);
}

void SetScriptHoldSeconds(float seconds) {
    g_script_hold_ticks = static_cast<uint64_t>(std::max(0.0f, seconds) * kFixedPhysicsHz);
}

int ControllerId() {
    return g_controller_id;
}

void GetRecentHistory(int steps, std::vector<float>* out) {
    if (out == nullptr || steps <= 0) {
        return;
    }
    out->assign(static_cast<size_t>(steps) * 6, 0.0f);
    const size_t available = std::min(g_history.size(), static_cast<size_t>(steps));
    for (size_t i = 0; i < available; ++i) {
        const std::array<float, 6>& snapshot = g_history[i];
        for (int j = 0; j < 6; ++j) {
            (*out)[i * 6 + j] = snapshot[j];
        }
    }
}

bool Enabled() {
    return g_enabled;
}

bool IsExternalController(int controller_id) {
    return g_enabled && controller_id == g_controller_id && controller_id > 0;
}

void ResetForEpisode() {
    g_move_x = 0.0f;
    g_move_y = 0.0f;
    for (auto& button : g_buttons) {
        button.second = false;
    }
    g_step_counter = 0;
    g_history.clear();
}

void SetMoveAxes(float move_x, float move_y) {
    g_move_x = move_x;
    g_move_y = move_y;
}

void SetButton(const std::string& name, bool held) {
    g_buttons[name] = held;
}

void GetCurrentAction(std::vector<float>* out) {
    if (out == nullptr) {
        return;
    }
    auto button_state = [](const std::string& name) -> float {
        auto it = g_buttons.find(name);
        return (it != g_buttons.end() && it->second) ? 1.0f : 0.0f;
    };
    *out = {g_move_x, g_move_y, button_state("jump"), button_state("crouch"),
            button_state("attack"), button_state("grab"), button_state("drop"),
            button_state("walk")};
}

void SetNativeReplayMode(bool enabled) {
    g_native_replay_mode = enabled;
}

void SetScriptStep(uint64_t step);  // forward decl; defined below, used by Apply() for the test harness

void Apply(Engine* engine) {
    if (!g_enabled || g_controller_id < 0) {
        return;
    }
    if (!g_script.empty() && !g_native_replay_mode) {
        // See rl_action.h's SetScriptPeriod comment -- g_step_counter is
        // ticks, the script's own "step" column is decisions, so divide
        // down to decision-cadence before looking up the held entry.
        SetScriptStep(g_step_counter / static_cast<uint64_t>(g_script_period));
    }
    ++g_step_counter;
    PlayerInput* controller = Input::Instance()->GetController(g_controller_id);
    if (controller == nullptr) {
        return;
    }

    // Camera-facing sync: find the character possessing this controller_id.
    // Training's controller 0 already owns a normal chase camera. A rendered
    // duel's controller 1 must instead get a private virtual camera: the
    // AngelScript attack target selector uses camera position/facing even for
    // a controlled character, and pointing the shared human camera at the
    // checkpoint would make attacks select/miss based on the human's view.
    // The virtual camera is never rendered or exposed to the human; it only
    // preserves the run15 camera-relative combat semantics while movement
    // remains body-relative in playercontrol.as.
    SceneGraph* scenegraph = engine->GetSceneGraph();
    if (scenegraph != nullptr) {
        for (Object* object : scenegraph->movement_objects_) {
            MovementObject* mo = static_cast<MovementObject*>(object);
            if (mo->controller_id == g_controller_id) {
                int camera_id = mo->camera_id;
                if (g_controller_id > 0) {
                    if (g_virtual_camera_id < 0) {
                        g_virtual_camera_id = ActiveCameras::Instance()->CreateVirtualCameraInstance();
                    }
                    camera_id = g_virtual_camera_id;
                    mo->camera_id = camera_id;
                }
                Camera* camera = ActiveCameras::GetCamera(camera_id);
                if (camera != nullptr) {
                    // mo->facing (the raw public field) is not reliably
                    // populated for a headless/never-rendered character;
                    // GetFacing() derives a proper unit vector from the
                    // rigged object's own animation rotation matrix
                    // (movementobject.cpp:1172-1174) and is always valid.
                    vec3 facing = mo->GetFacing();
                    facing.y() = 0.0f;
                    if (length_squared(facing) < 0.0001f) {
                        facing = vec3(0.0f, 0.0f, 1.0f);
                    } else {
                        facing = normalize(facing);
                    }
                    camera->SetFlatFacing(facing);
                    if (g_controller_id > 0) {
                        camera->SetFacing(facing);
                        camera->SetPos(mo->position + vec3(0.0f, 1.0f, 0.0f));
                    }
                }
                break;
            }
        }
    }

    // Movement axes: AS_GetMoveXAxis/AS_GetMoveYAxis (Source/Scripting/angelscript/
    // asfuncs.cpp) read key_down["right"].depth - key_down["left"].depth and
    // key_down["down"].depth - key_down["up"].depth respectively. playercontrol.as's
    // GetTargetVelocity() then does `target_velocity -= MoveYAxis * facing`, so a
    // NEGATIVE MoveYAxis (i.e. "up" held, since MoveYAxis = down - up) is what
    // moves the character forward -- matches this header's "y = forward(+)/back(-)"
    // contract, which is why "up" (not "down") is keyed off g_move_y > 0 below.
    ApplyKey(*controller, "right", g_move_x > 0.0f, g_move_x);
    ApplyKey(*controller, "left", g_move_x < 0.0f, -g_move_x);
    ApplyKey(*controller, "up", g_move_y > 0.0f, g_move_y);
    ApplyKey(*controller, "down", g_move_y < 0.0f, -g_move_y);

    for (const auto& button : g_buttons) {
        ApplyKey(*controller, button.first, button.second, button.second ? 1.0f : 0.0f);
    }

    // Record this step's snapshot for RLObservation's action-history block.
    // Read back from g_buttons (default false for a button never touched)
    // rather than re-deriving from the controller's key_down state, so this
    // records intent (what RLAction was asked to apply) not an inferred
    // reconstruction of it.
    auto button_state = [](const std::string& name) -> float {
        auto it = g_buttons.find(name);
        return (it != g_buttons.end() && it->second) ? 1.0f : 0.0f;
    };
    g_history.push_front({g_move_x, g_move_y, button_state("jump"), button_state("crouch"),
                           button_state("attack"), button_state("grab")});
    while (g_history.size() > kMaxHistory) {
        g_history.pop_back();
    }
}

bool ScriptFinished() {
    if (g_script.empty()) {
        return false;
    }
    const uint64_t decision_index = g_step_counter / static_cast<uint64_t>(g_script_period);
    return decision_index > g_script.back().step;
}

int64_t TicksSinceScriptFinished() {
    if (!ScriptFinished()) {
        return -1;
    }
    // The first tick where decision_index (g_step_counter / period) exceeds
    // the last entry's step -- i.e. exactly where ScriptFinished() first went
    // true -- computed directly rather than tracked with extra mutable state,
    // since it's a pure function of the script data + period already held.
    const uint64_t finish_tick = (g_script.back().step + 1) * static_cast<uint64_t>(g_script_period);
    return (g_step_counter > finish_tick) ? static_cast<int64_t>(g_step_counter - finish_tick) : 0;
}

int64_t SettleTicks() {
    // OGRL-20260819-039: keep simulating this long after the script ends so
    // the final recorded action's CONSEQUENCE is visible before freezing.
    // Sized from measurement, not taste: on tape 1905_w0_e58 the script ends
    // at tick 232 and the knockout registers at tick 234, after which the
    // ragdoll needs a beat to read on screen. 0.75s = 90 ticks at 120Hz
    // covers that comfortably while staying far short of the ~2s it takes the
    // arena's own round logic to intervene.
    return static_cast<int64_t>(0.75f * kFixedPhysicsHz);
}

bool ReadyToQuit() {
    const int64_t since = TicksSinceScriptFinished();
    if (since < 0) {
        return false;
    }
    return static_cast<uint64_t>(since) >= g_script_hold_ticks;
}

bool LoadScript(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    g_script.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }
        if (fields.size() != 9) {
            continue;  // malformed line, skip rather than fail the whole script
        }
        ScriptEntry entry;
        entry.step = static_cast<uint64_t>(std::stoull(fields[0]));
        entry.move_x = std::stof(fields[1]);
        entry.move_y = std::stof(fields[2]);
        entry.jump = std::stoi(fields[3]) != 0;
        entry.crouch = std::stoi(fields[4]) != 0;
        entry.attack = std::stoi(fields[5]) != 0;
        entry.grab = std::stoi(fields[6]) != 0;
        entry.drop = std::stoi(fields[7]) != 0;
        entry.walk = std::stoi(fields[8]) != 0;
        g_script.push_back(entry);
    }
    std::sort(g_script.begin(), g_script.end(), [](const ScriptEntry& a, const ScriptEntry& b) { return a.step < b.step; });
    return true;
}

void SetScriptStep(uint64_t step) {
    if (g_script.empty()) {
        return;
    }
    // Held-until-next-line semantics: find the last entry at or before `step`.
    const ScriptEntry* active = nullptr;
    for (const auto& entry : g_script) {
        if (entry.step > step) {
            break;
        }
        active = &entry;
    }
    if (active == nullptr) {
        return;
    }
    SetMoveAxes(active->move_x, active->move_y);
    SetButton("jump", active->jump);
    SetButton("crouch", active->crouch);
    SetButton("attack", active->attack);
    SetButton("grab", active->grab);
    SetButton("drop", active->drop);
    SetButton("walk", active->walk);
}

}  // namespace RLAction
