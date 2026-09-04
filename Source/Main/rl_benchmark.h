// Lightweight, opt-in benchmark instrumentation for production engine steps.
#pragma once

#include <cstdint>
#include <string>

class Engine;

namespace RLBenchmark {

// measure_window_seconds > 0 switches the measurement stop condition from a
// fixed step count to a fixed wall-clock duration (measure_steps is then a
// safety cap only). barrier_workers > 0 (with a non-empty barrier_dir) makes
// this process block, once warmup completes, until barrier_workers processes
// have all reached the same point -- so concurrently-launched workers begin
// their measurement window at (approximately) the same wall-clock instant.
// See research-artifacts/implementation_plan_m4_gym.md Stage 0.3.
// progress_interval_seconds > 0 (Stage 0.8) emits an "RL_BENCHMARK_PROGRESS"
// JSON line to stdout every ~progress_interval_seconds of measurement, giving
// a bucketed steps/s time series across a long sustained run without needing
// to restart the process (which would reload the level and reset thermal
// state each time).
// reset_after_warmup (Stage 4, Approach B) triggers exactly one in-process
// Engine::ResetRLTrainingScenario() call at the warmup/measurement boundary,
// then measures the steps immediately following it -- so a single run
// reports reset latency for that one reset. See Tools/rl/reset_bakeoff.py.
void Configure(bool enabled, uint64_t warmup_steps, uint64_t measure_steps, unsigned int seed,
               double measure_window_seconds = 0.0,
               const std::string& barrier_dir = std::string(),
               int barrier_workers = 0,
               double progress_interval_seconds = 0.0,
               bool reset_after_warmup = false);
bool Enabled();
bool ResetAfterWarmup();
unsigned int Seed();

void OnEngineInitialized();
void OnShaderPreloadStarted();
void OnShaderPreloadFinished();
void OnLevelLoaded();

// Returns a bounded number of exact fixed timesteps for the next outer engine
// update, or -1 while normal wall-clock scheduling should remain active.
int ManualStepCount();

// Returns true when the requested measurement window has completed.
bool OnTimestepComplete(Engine* engine);
void Report();

}  // namespace RLBenchmark
