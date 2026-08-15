// Lightweight, opt-in benchmark instrumentation for production engine steps.
#include "rl_benchmark.h"

#include <Main/engine.h>

#include <chrono>
#include <iomanip>
#include <iostream>

namespace RLBenchmark {
namespace {

using Clock = std::chrono::steady_clock;

struct State {
    bool enabled = false;
    bool level_loaded = false;
    bool measuring = false;
    bool completed = false;
    bool reported = false;
    uint64_t warmup_steps = 0;
    uint64_t measure_steps = 0;
    uint64_t completed_steps = 0;
    uint64_t measured_steps = 0;
    unsigned int seed = 0;
    double engine_initialize_seconds = 0.0;
    double shader_preload_seconds = 0.0;
    double level_load_seconds = 0.0;
    double measurement_seconds = 0.0;
    Clock::time_point configured_at;
    Clock::time_point engine_initialized_at;
    Clock::time_point shader_preload_started_at;
    Clock::time_point measurement_started_at;
};

State state;

double SecondsSince(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

}  // namespace

void Configure(bool enabled, uint64_t warmup_steps, uint64_t measure_steps, unsigned int seed) {
    state = State();
    state.enabled = enabled;
    state.warmup_steps = warmup_steps;
    state.measure_steps = measure_steps;
    state.seed = seed;
    state.configured_at = Clock::now();
}

bool Enabled() {
    return state.enabled;
}

unsigned int Seed() {
    return state.seed;
}

void OnEngineInitialized() {
    if (!state.enabled) {
        return;
    }
    state.engine_initialize_seconds = SecondsSince(state.configured_at);
    state.engine_initialized_at = Clock::now();
}

void OnShaderPreloadStarted() {
    if (state.enabled) {
        state.shader_preload_started_at = Clock::now();
    }
}

void OnShaderPreloadFinished() {
    if (state.enabled) {
        state.shader_preload_seconds = SecondsSince(state.shader_preload_started_at);
    }
}

void OnLevelLoaded() {
    if (!state.enabled || state.level_loaded) {
        return;
    }
    state.level_loaded = true;
    state.level_load_seconds = SecondsSince(state.engine_initialized_at);
}

bool OnTimestepComplete(Engine* engine) {
    if (!state.enabled || !state.level_loaded || state.completed) {
        return false;
    }

    ++state.completed_steps;
    if (!state.measuring && state.completed_steps <= state.warmup_steps) {
        return false;
    }
    if (!state.measuring) {
        state.measuring = true;
        state.measurement_started_at = Clock::now();
    }

    if (state.measuring) {
        ++state.measured_steps;
        if (state.measured_steps >= state.measure_steps) {
            state.measurement_seconds = SecondsSince(state.measurement_started_at);
            state.completed = true;
            engine->quitting_ = true;
            return true;
        }
    }
    return false;
}

void Report() {
    if (!state.enabled || state.reported) {
        return;
    }
    state.reported = true;
    const double steps_per_second = state.measurement_seconds > 0.0
                                        ? static_cast<double>(state.measured_steps) / state.measurement_seconds
                                        : 0.0;
    std::cout << std::fixed << std::setprecision(6)
              << "RL_BENCHMARK_RESULT {"
              << "\"benchmark_completed\":" << (state.completed ? 1 : 0) << ','
              << "\"seed\":" << state.seed << ','
              << "\"fixed_physics_hz\":120,"
              << "\"warmup_steps\":" << state.warmup_steps << ','
              << "\"measured_steps\":" << state.measured_steps << ','
              << "\"steps_per_second\":" << steps_per_second << ','
              << "\"measurement_seconds\":" << state.measurement_seconds << ','
              << "\"engine_initialize_seconds\":" << state.engine_initialize_seconds << ','
              << "\"level_load_seconds\":" << state.level_load_seconds << ','
              << "\"shader_preload_seconds\":" << state.shader_preload_seconds
              << "}" << std::endl;
}

}  // namespace RLBenchmark
