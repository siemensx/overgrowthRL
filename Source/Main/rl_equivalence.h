// Stage 1 production-equivalence replay comparator: opt-in per-step state
// digest + hash chain, and (scoped-down v1) legal-input trace record/replay.
// See research-artifacts/implementation_plan_m4_gym.md Stage 1.
//
// Scope note (logged in research-log OGRL-20260815-035): none of the current
// benchmark scenarios (duel/four_character/six_character) have a genuinely
// `controlled`-with-live-input character -- the "PC slot" character is
// script-path-matched to run at update_script_period=1 but is still driven by
// AI logic, not live input, in headless mode. The input-trace mechanism below
// is real and hooked at the point the plan specifies (UpdateControls), but is
// currently exercised on an empty trace (0 controlled characters) until
// Stage 5 adds an actual controllable agent. The state digest is the part of
// Stage 1 that is fully exercised today and is what gates Stage 2/3.
#pragma once

#include <cstdint>
#include <string>

class Engine;
class SceneGraph;

namespace RLEquivalence {

enum class Mode {
    kOff,
    kRecord,
    kReplay,
};

// digest_path: newline-delimited JSON, one line per physics step, each line
// containing the full per-character raw quantity record plus the running
// FNV-1a-64 hash chain value after that step (chain[i] = FNV(chain[i-1] ++
// serialize(step_i))). trace_path (may be empty): legal-input trace output
// for controller_id-addressed input state entering UpdateControls.
void Configure(Mode mode, const std::string& digest_path, const std::string& trace_path,
               unsigned int initial_seed = 0);
// Configure an online replay against a per-episode native digest.  The
// engine remains the authority: it consumes the recorded legal actions and
// compares its own post-tick state-chain to the expected chain before the
// rendered window is allowed to present the result as exact.
void ConfigureReplay(const std::string& expected_path, const std::string& report_path);
void ApplyReplayAction(uint64_t tick_index);
bool ReplayExhausted();
bool Enabled();

// Marks the canonical episode boundary and resets the per-episode state hash
// chain.  Training calls this after each shm reset; deterministic rendered
// replay calls it after RLReplaySeed has applied the archived scenario.
void OnEpisodeReset(unsigned int seed, int reset_mode, float difficulty, int opponents,
                    float weapons, int species);

// Called once per physics step from the same site as RLBenchmark::OnTimestepComplete.
// Maintains its own internal step counter (reset by Configure()).
void OnTimestepComplete(Engine* engine);

// Called from Engine::UpdateControls, once per physics step, before script
// logic consumes input for the step.
void OnUpdateControls(Engine* engine);

// Flushes buffered output and writes the trace/digest file headers+footers.
void Finalize();

}  // namespace RLEquivalence
