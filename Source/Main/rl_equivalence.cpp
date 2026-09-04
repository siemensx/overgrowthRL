#include "rl_equivalence.h"

#include "rl_action.h"

#include <Main/engine.h>
#include <Main/scenegraph.h>
#include <Objects/movementobject.h>
#include <Objects/riggedobject.h>
#include <Graphics/animationclient.h>
#include <Graphics/camera.h>
#include <Math/rng_streams.h>

#include <cstdio>
#include <cstdlib>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace RLEquivalence {
namespace {

Mode g_mode = Mode::kOff;
std::string g_digest_path;
std::string g_trace_path;
std::ofstream g_digest_file;
std::ofstream g_trace_file;
std::ofstream g_report_file;
std::ofstream g_observed_file;
uint64_t g_hash_chain = 0xcbf29ce484222325ULL;  // FNV-1a-64 offset basis
uint64_t g_controlled_input_records = 0;
uint64_t g_digest_step_counter = 0;
uint64_t g_control_step_counter = 0;
std::vector<uint64_t> g_expected_chains;
std::vector<std::array<float, 8>> g_expected_actions;
bool g_replay_diverged = false;
uint64_t g_replay_first_tick = 0;
uint64_t g_replay_expected_chain = 0;
uint64_t g_replay_actual_chain = 0;

constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;

// FNV-1a-64, folded into the running chain: chain = FNV(chain_bytes ++ data).
void FoldIntoChain(const std::string& data) {
    uint64_t h = g_hash_chain;
    for (unsigned char c : data) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    g_hash_chain = h;
}

std::string FloatStr(float v) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(8) << v;
    return out.str();
}

// primary_weapon_slot / weapon_slots are real AngelScript globals in
// Data/Scripts/aschar.as (verified: lines 401, 404). temp_health,
// permanent_health, blood_health, block_health, knocked_out, state are also
// verified real globals there (lines 160-169, 9832-9847). This is a scoped
// subset of the plan's full Stage 1.3 enumeration -- not yet included: full
// per-physics-bone transforms (Skeleton::physics_bones), the ordered Bullet
// contact set across all three worlds, and per-attack-event firing/damage
// records. Logged explicitly as deferred in research-log OGRL-20260815-035.
std::string SerializeCharacter(MovementObject* mo) {
    std::ostringstream out;
    out << "{\"id\":" << mo->GetID()
        << ",\"controlled\":" << (mo->controlled ? 1 : 0)
        << ",\"controller_id\":" << mo->controller_id
        << ",\"is_player\":" << (mo->is_player ? 1 : 0)
        << ",\"remote\":" << (mo->remote ? 1 : 0)
        << ",\"pos\":[" << FloatStr(mo->position.x()) << ',' << FloatStr(mo->position.y()) << ',' << FloatStr(mo->position.z()) << ']'
        << ",\"vel\":[" << FloatStr(mo->velocity.x()) << ',' << FloatStr(mo->velocity.y()) << ',' << FloatStr(mo->velocity.z()) << ']'
        // mo->facing (the raw public field) reads as [0,0,0] for a never-rendered
        // headless character -- GetFacing() derives a proper unit vector from the
        // rigged object's own animation rotation matrix instead. Same bug, same
        // fix as RLAction::Apply()'s camera-facing sync (OGRL-20260816-005);
        // fixing it here too rather than leaving this digest field permanently
        // degenerate. Recorded digests from before this fix are not comparable
        // against digests recorded after it for the facing field specifically
        // (their hash chains will legitimately differ) -- that's this fix
        // working, not a new divergence.
        << ",\"facing\":[" << FloatStr(mo->GetFacing().x()) << ',' << FloatStr(mo->GetFacing().y()) << ',' << FloatStr(mo->GetFacing().z()) << ']';

    RiggedObject* rigged = mo->rigged_object();
    if (rigged != nullptr) {
        vec3 angular_velocity = rigged->GetAvgAngularVelocity();
        out << ",\"ang_vel\":[" << FloatStr(angular_velocity.x()) << ',' << FloatStr(angular_velocity.y()) << ',' << FloatStr(angular_velocity.z()) << ']'
            << ",\"anim\":\"" << rigged->GetAnimClient().GetCurrAnim() << '"'
            << ",\"anim_phase\":" << FloatStr(rigged->GetAnimClient().GetNormalizedAnimTime());
    } else {
        out << ",\"ang_vel\":[0,0,0],\"anim\":\"\",\"anim_phase\":0";
    }

    out << ",\"temp_health\":" << FloatStr(mo->ASGetFloatVar("temp_health"))
        << ",\"permanent_health\":" << FloatStr(mo->ASGetFloatVar("permanent_health"))
        << ",\"blood_health\":" << FloatStr(mo->ASGetFloatVar("blood_health"))
        << ",\"block_health\":" << FloatStr(mo->ASGetFloatVar("block_health"))
        << ",\"knocked_out\":" << mo->ASGetIntVar("knocked_out")
        << ",\"state\":" << mo->ASGetIntVar("state");

    const int primary_slot = mo->ASGetIntVar("primary_weapon_slot");
    const int primary_weapon_item_id = mo->ASGetArrayIntVar("weapon_slots", primary_slot);
    out << ",\"primary_weapon_item_id\":" << primary_weapon_item_id;

    // roll_count / active_blocking added for Stage 5.3's roll/block validation
    // (research-log OGRL-20260816-008) -- both plain script globals
    // (Data/Scripts/aschar.as:49,277), read the same way as everything above.
    // Widens the equivalence digest's coverage; a strict-mode comparison
    // against a digest recorded before this change will legitimately show a
    // different hash chain -- that's this addition working, not a regression.
    out << ",\"roll_count\":" << mo->ASGetIntVar("roll_count")
        << ",\"active_blocking\":" << (mo->ASGetBoolVar("active_blocking") ? 1 : 0);

    out << '}';
    return out.str();
}

std::string SerializeState(Engine* engine, uint64_t step_index) {
    SceneGraph* scenegraph = engine->GetSceneGraph();
    std::ostringstream line;
    line << "{\"kind\":\"tick\",\"step\":" << step_index
         << ",\"rng_gameplay_draws\":" << RngStreams::DrawCount(RngStreams::Stream::kGameplay)
         << ",\"rng_cosmetic_draws\":" << RngStreams::DrawCount(RngStreams::Stream::kCosmetic);
    Camera* diag_camera = ActiveCameras::GetCamera(0);
    vec3 diag_cam_pos = diag_camera != nullptr ? diag_camera->GetPos() : vec3(-9999.0f);
    line << ",\"diag_cam_pos\":[" << FloatStr(diag_cam_pos.x()) << ',' << FloatStr(diag_cam_pos.y()) << ',' << FloatStr(diag_cam_pos.z()) << ']';
    std::vector<float> action;
    RLAction::GetCurrentAction(&action);
    line << ",\"action\":[";
    for (size_t i = 0; i < action.size(); ++i) {
        if (i != 0) line << ',';
        line << FloatStr(action[i]);
    }
    line << "],\"characters\":[";
    bool first = true;
    if (scenegraph != nullptr) {
        for (Object* object : scenegraph->movement_objects_) {
            MovementObject* mo = static_cast<MovementObject*>(object);
            if (!first) line << ',';
            first = false;
            line << SerializeCharacter(mo);
        }
    }
    line << "]}";
    return line.str();
}

bool ParseChain(const std::string& line, uint64_t* chain) {
    const std::string marker = ",\"chain\":";
    const size_t start = line.find(marker);
    if (start == std::string::npos || chain == nullptr) return false;
    const char* begin = line.c_str() + start + marker.size();
    char* end = nullptr;
    unsigned long long value = std::strtoull(begin, &end, 10);
    if (end == begin) return false;
    *chain = static_cast<uint64_t>(value);
    return true;
}

bool ParseAction(const std::string& line, std::array<float, 8>* action) {
    const std::string marker = "\"action\":[";
    const size_t start = line.find(marker);
    if (start == std::string::npos || action == nullptr) return false;
    const char* cursor = line.c_str() + start + marker.size();
    for (size_t i = 0; i < action->size(); ++i) {
        char* end = nullptr;
        (*action)[i] = std::strtof(cursor, &end);
        if (end == cursor) return false;
        cursor = end;
        if (i + 1 < action->size()) {
            if (*cursor != ',') return false;
            ++cursor;
        }
    }
    return true;
}

}  // namespace

void Configure(Mode mode, const std::string& digest_path, const std::string& trace_path,
               unsigned int initial_seed) {
    g_mode = mode;
    g_digest_path = digest_path;
    g_trace_path = trace_path;
    g_hash_chain = kFnvOffset;
    g_controlled_input_records = 0;
    g_digest_step_counter = 0;
    g_control_step_counter = 0;
    g_expected_chains.clear();
    g_expected_actions.clear();
    g_replay_diverged = false;
    g_replay_first_tick = 0;
    g_replay_expected_chain = 0;
    g_replay_actual_chain = 0;
    if (g_report_file.is_open()) g_report_file.close();
    if (g_observed_file.is_open()) g_observed_file.close();
    if (g_mode == Mode::kRecord && !g_digest_path.empty()) {
        g_digest_file.open(g_digest_path, std::ios::out | std::ios::trunc);
    }
    if (g_mode == Mode::kRecord && !g_trace_path.empty()) {
        g_trace_file.open(g_trace_path, std::ios::out | std::ios::trunc);
    }
    if (g_mode == Mode::kRecord && !g_digest_path.empty()) {
        // The first marker is the natural initial episode. Later markers are
        // emitted by RLShmTransport after each canonical reset.
        OnEpisodeReset(initial_seed, 0, -1.0f, -1, -1.0f, -1);
    }
}

void ConfigureReplay(const std::string& expected_path, const std::string& report_path) {
    Configure(Mode::kReplay, "", "");
    g_digest_path = expected_path;
    g_trace_path = report_path;
    std::ifstream expected(expected_path);
    std::string line;
    while (std::getline(expected, line)) {
        if (line.find("\"kind\":\"tick\"") == std::string::npos) continue;
        uint64_t chain = 0;
        std::array<float, 8> action{};
        if (ParseChain(line, &chain) && ParseAction(line, &action)) {
            g_expected_chains.push_back(chain);
            g_expected_actions.push_back(action);
        }
    }
    if (!report_path.empty()) g_report_file.open(report_path, std::ios::out | std::ios::trunc);
    if (!report_path.empty()) {
        g_observed_file.open(report_path + ".actual.jsonl", std::ios::out | std::ios::trunc);
    }
}

void ApplyReplayAction(uint64_t tick_index) {
    if (g_mode != Mode::kReplay || tick_index >= g_expected_actions.size()) return;
    const std::array<float, 8>& action = g_expected_actions[tick_index];
    RLAction::SetMoveAxes(action[0], action[1]);
    RLAction::SetButton("jump", action[2] > 0.5f);
    RLAction::SetButton("crouch", action[3] > 0.5f);
    RLAction::SetButton("attack", action[4] > 0.5f);
    RLAction::SetButton("grab", action[5] > 0.5f);
    RLAction::SetButton("drop", action[6] > 0.5f);
    RLAction::SetButton("walk", action[7] > 0.5f);
}

bool ReplayExhausted() {
    return g_mode == Mode::kReplay && !g_expected_chains.empty() &&
           g_digest_step_counter >= g_expected_chains.size();
}

bool Enabled() {
    return g_mode != Mode::kOff;
}

void OnEpisodeReset(unsigned int seed, int reset_mode, float difficulty, int opponents,
                    float weapons, int species) {
    if (g_mode == Mode::kOff || g_mode == Mode::kReplay) return;
    g_hash_chain = kFnvOffset;
    g_digest_step_counter = 0;
    g_control_step_counter = 0;
    if (g_digest_file.is_open()) {
        g_digest_file << "{\"kind\":\"reset\",\"seed\":" << seed
                      << ",\"reset_mode\":" << reset_mode
                      << ",\"difficulty\":" << FloatStr(difficulty)
                      << ",\"opponents\":" << opponents
                      << ",\"weapons\":" << FloatStr(weapons)
                      << ",\"species\":" << species << "}\n";
        g_digest_file.flush();
    }
}

void OnTimestepComplete(Engine* engine) {
    if (g_mode == Mode::kOff) {
        return;
    }
    SceneGraph* scenegraph = engine->GetSceneGraph();
    if (scenegraph == nullptr) {
        return;
    }
    const uint64_t step_index = g_digest_step_counter++;

    std::ostringstream line;
    // Stage 2 (research-log OGRL-20260815-036): per-episode draw counters per
    // stream, "far more sensitive than catching the resulting behavioral
    // divergence" -- a single extra/missing draw shows up here immediately,
    // before it has a chance to compound into a visible trajectory difference.
    // Diagnostic field kept permanently (research-log OGRL-20260815-038):
    // level.as::SetAnimUpdateFreqs() reads camera.GetPos() (camera ID 0,
    // since level.as has no owning character) to weight each character's
    // animation-update budget -- AGENTS.md/the plan flagged this as a live
    // correctness hazard in headless mode. Investigated as the leading
    // suspect for the step-975 divergence; ruled out (verified bit-identical
    // across diverging runs -- the real cause was ASLR, see noaslr.py). Left
    // in the digest since it is cheap and directly answers "is the phantom
    // camera moving," which remains relevant for Stage 3a.
    const std::string serialized = SerializeState(engine, step_index);
    FoldIntoChain(serialized);

    if (g_digest_file.is_open()) {
        g_digest_file << serialized.substr(0, serialized.size() - 1) << ",\"chain\":" << g_hash_chain << "}\n";
        g_digest_file.flush();
    }
    if (g_mode == Mode::kReplay) {
        if (g_observed_file.is_open()) {
            g_observed_file << serialized.substr(0, serialized.size() - 1) << ",\"chain\":" << g_hash_chain << "}\n";
            g_observed_file.flush();
        }
        if (step_index >= g_expected_chains.size()) {
            if (!g_replay_diverged) {
                g_replay_diverged = true;
                g_replay_first_tick = step_index;
                g_replay_expected_chain = 0;
                g_replay_actual_chain = g_hash_chain;
            }
        } else if (!g_replay_diverged && g_hash_chain != g_expected_chains[step_index]) {
            g_replay_diverged = true;
            g_replay_first_tick = step_index;
            g_replay_expected_chain = g_expected_chains[step_index];
            g_replay_actual_chain = g_hash_chain;
        }
        ApplyReplayAction(step_index + 1);
    }
}

void OnUpdateControls(Engine* engine) {
    if (g_mode == Mode::kOff) {
        return;
    }
    SceneGraph* scenegraph = engine->GetSceneGraph();
    if (scenegraph == nullptr) {
        return;
    }
    const uint64_t step_index = g_control_step_counter++;
    // Per Stage 1.1: capture the legal input vector (controller_id + button/axis
    // state) for every genuinely controlled character. In the current
    // AI-vs-AI benchmark scenarios this set is empty -- verified by construction,
    // not assumed: GetControlledMovementObjects() only returns characters with
    // live input possession (Source/Game/avatar_control_manager.cpp), which
    // headless mode never grants. This records that fact per step (an explicit
    // empty record, not a silent gap) so the comparator can assert trace step
    // count matches the run's step count per Stage 1.2, and the mechanism is
    // exercised for real once Stage 5 adds a controllable agent.
    std::vector<MovementObject*> controlled = scenegraph->GetControlledMovementObjects();
    if (g_trace_file.is_open()) {
        g_trace_file << "{\"step\":" << step_index << ",\"controlled_count\":" << controlled.size() << "}\n";
    }
    g_controlled_input_records += controlled.size();
}

void Finalize() {
    if (g_mode == Mode::kOff) {
        return;
    }
    if (g_digest_file.is_open()) {
        g_digest_file.close();
    }
    if (g_trace_file.is_open()) {
        g_trace_file.close();
    }
    if (g_observed_file.is_open()) {
        g_observed_file.close();
    }
    if (g_mode == Mode::kReplay && g_report_file.is_open()) {
        const bool exact = !g_replay_diverged && g_digest_step_counter == g_expected_chains.size();
        g_report_file << "{\"verification\":\"" << (exact ? "exact_simulation_verified" : "diverged")
                      << "\",\"ticks\":" << g_digest_step_counter
                      << ",\"expected_ticks\":" << g_expected_chains.size();
        if (g_replay_diverged) {
            g_report_file << ",\"tick\":" << g_replay_first_tick
                          << ",\"expected_chain\":" << g_replay_expected_chain
                          << ",\"actual_chain\":" << g_replay_actual_chain;
        }
        g_report_file << "}\n";
        g_report_file.flush();
        g_report_file.close();
    }
    std::cout << std::fixed << std::setprecision(6)
              << "RL_EQUIVALENCE_RESULT {"
              << "\"total_steps\":" << g_digest_step_counter << ','
              << "\"final_hash_chain\":" << g_hash_chain << ','
              << "\"controlled_input_records\":" << g_controlled_input_records << ','
              << "\"digest_path\":\"" << g_digest_path << "\","
              << "\"trace_path\":\"" << g_trace_path << "\""
              << "}" << std::endl;
}

}  // namespace RLEquivalence
