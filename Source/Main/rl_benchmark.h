// Lightweight, opt-in benchmark instrumentation for production engine steps.
#pragma once

#include <cstdint>

class Engine;

namespace RLBenchmark {

void Configure(bool enabled, uint64_t warmup_steps, uint64_t measure_steps, unsigned int seed);
bool Enabled();
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
