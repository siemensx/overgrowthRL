// Lightweight, opt-in benchmark instrumentation for production engine steps.
#include "rl_benchmark.h"

#include <Main/engine.h>
#include <Objects/object.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace RLBenchmark {
namespace {

using Clock = std::chrono::steady_clock;

// Six 120 Hz physics steps are one 20 Hz policy interval. Keeping the manual
// batch to one control interval amortizes outer-loop services without starving
// input/event handling for long simulation bursts.
const uint64_t kManualStepBatchSize = 6;

struct SceneSnapshot {
    bool valid = false;
    uint64_t hash = 0;
    uint64_t object_count = 0;
    uint64_t movement_object_count = 0;
};

void HashBytes(uint64_t* hash, const void* data, size_t size) {
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        *hash ^= bytes[i];
        *hash *= 1099511628211ULL;
    }
}

void HashUInt64(uint64_t* hash, uint64_t value) {
    HashBytes(hash, &value, sizeof(value));
}

void HashFloat(uint64_t* hash, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    HashBytes(hash, &bits, sizeof(bits));
}

void HashString(uint64_t* hash, const std::string& value) {
    HashUInt64(hash, value.size());
    HashBytes(hash, value.data(), value.size());
}

SceneSnapshot CaptureSceneSnapshot(Engine* engine) {
    SceneSnapshot snapshot;
    SceneGraph* scenegraph = engine->GetSceneGraph();
    if (scenegraph == NULL) {
        return snapshot;
    }

    snapshot.valid = true;
    snapshot.hash = 1469598103934665603ULL;
    snapshot.object_count = scenegraph->objects_.size();
    snapshot.movement_object_count = scenegraph->movement_objects_.size();
    HashUInt64(&snapshot.hash, snapshot.object_count);
    HashUInt64(&snapshot.hash, snapshot.movement_object_count);

    for (const Object* object : scenegraph->objects_) {
        HashUInt64(&snapshot.hash, static_cast<uint64_t>(object->GetID()));
        HashUInt64(&snapshot.hash, static_cast<uint64_t>(object->GetType()));
        HashUInt64(&snapshot.hash, object->enabled_ ? 1 : 0);
        HashUInt64(&snapshot.hash, object->collidable ? 1 : 0);
        HashString(&snapshot.hash, object->name);
        HashString(&snapshot.hash, object->obj_file);

        const vec3& translation = object->GetTranslation();
        const vec3& scale = object->GetScale();
        const quaternion& rotation = object->GetRotation();
        for (int i = 0; i < 3; ++i) {
            HashFloat(&snapshot.hash, translation[i]);
            HashFloat(&snapshot.hash, scale[i]);
        }
        for (int i = 0; i < 4; ++i) {
            HashFloat(&snapshot.hash, rotation[i]);
        }

        HashUInt64(&snapshot.hash, object->connected_from.size());
        for (int id : object->connected_from) {
            HashUInt64(&snapshot.hash, static_cast<uint64_t>(id));
        }
        HashUInt64(&snapshot.hash, object->connected_to.size());
        for (int id : object->connected_to) {
            HashUInt64(&snapshot.hash, static_cast<uint64_t>(id));
        }
    }
    return snapshot;
}

struct State {
    bool enabled = false;
    bool level_loaded = false;
    bool measuring = false;
    bool completed = false;
    bool reported = false;
    bool reset_after_warmup = false;
    bool reset_attempted = false;
    bool reset_in_progress = false;
    bool reset_succeeded = false;
    bool reset_state_match = false;
    uint64_t warmup_steps = 0;
    uint64_t measure_steps = 0;
    uint64_t completed_steps = 0;
    uint64_t measured_steps = 0;
    unsigned int seed = 0;
    double engine_initialize_seconds = 0.0;
    double shader_preload_seconds = 0.0;
    double level_load_seconds = 0.0;
    double reset_latency_seconds = 0.0;
    double measurement_seconds = 0.0;
    Clock::time_point configured_at;
    Clock::time_point engine_initialized_at;
    Clock::time_point shader_preload_started_at;
    Clock::time_point measurement_started_at;
    SceneSnapshot initial_snapshot;
    SceneSnapshot reset_snapshot;
};

State state;

double SecondsSince(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

}  // namespace

void Configure(bool enabled, uint64_t warmup_steps, uint64_t measure_steps, unsigned int seed, bool reset_after_warmup) {
    state = State();
    state.enabled = enabled;
    state.warmup_steps = warmup_steps;
    state.measure_steps = measure_steps;
    state.seed = seed;
    state.reset_after_warmup = reset_after_warmup;
    state.configured_at = Clock::now();
}

bool Enabled() {
    return state.enabled;
}

bool ResetAfterWarmup() {
    return state.enabled && state.reset_after_warmup;
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
    if (!state.enabled) {
        return;
    }
    if (!state.level_loaded) {
        state.level_loaded = true;
        state.level_load_seconds = SecondsSince(state.engine_initialized_at);
        if (state.reset_after_warmup) {
            state.initial_snapshot = CaptureSceneSnapshot(Engine::Instance());
        }
    } else if (state.reset_in_progress) {
        state.reset_snapshot = CaptureSceneSnapshot(Engine::Instance());
    }
}

int ManualStepCount() {
    if (!state.enabled || !state.level_loaded || state.completed) {
        return -1;
    }

    const uint64_t total_steps = state.warmup_steps + state.measure_steps;
    const uint64_t remaining_steps = total_steps - state.completed_steps;
    return static_cast<int>(remaining_steps < kManualStepBatchSize ? remaining_steps : kManualStepBatchSize);
}

bool OnTimestepComplete(Engine* engine) {
    if (!state.enabled || !state.level_loaded || state.completed) {
        return false;
    }

    ++state.completed_steps;
    if (state.reset_after_warmup && !state.reset_attempted && state.completed_steps == state.warmup_steps) {
        state.reset_attempted = true;
        state.reset_in_progress = true;
        const Clock::time_point reset_started_at = Clock::now();
        state.reset_succeeded = engine->ResetRLTrainingScenario(state.seed);
        state.reset_latency_seconds = SecondsSince(reset_started_at);
        state.reset_in_progress = false;
        state.reset_state_match = state.reset_succeeded && state.initial_snapshot.valid && state.reset_snapshot.valid &&
                                  state.initial_snapshot.hash == state.reset_snapshot.hash &&
                                  state.initial_snapshot.object_count == state.reset_snapshot.object_count &&
                                  state.initial_snapshot.movement_object_count == state.reset_snapshot.movement_object_count;
        if (!state.reset_succeeded || !state.reset_state_match) {
            engine->quitting_ = true;
        }
        // The reload destroys the old SceneGraph. End this outer update before
        // any code can reuse locals captured from the pre-reset world.
        return true;
    }
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
              << "\"reset_attempted\":" << (state.reset_attempted ? 1 : 0) << ','
              << "\"reset_succeeded\":" << (state.reset_succeeded ? 1 : 0) << ','
              << "\"reset_state_match\":" << (state.reset_state_match ? 1 : 0) << ','
              << "\"reset_latency_ms\":" << state.reset_latency_seconds * 1000.0 << ','
              << "\"initial_state_hash\":" << state.initial_snapshot.hash << ','
              << "\"reset_state_hash\":" << state.reset_snapshot.hash << ','
              << "\"initial_object_count\":" << state.initial_snapshot.object_count << ','
              << "\"reset_object_count\":" << state.reset_snapshot.object_count << ','
              << "\"shader_preload_seconds\":" << state.shader_preload_seconds
              << "}" << std::endl;
}

}  // namespace RLBenchmark
