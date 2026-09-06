// Lightweight, opt-in benchmark instrumentation for production engine steps.
#include "rl_benchmark.h"

#include <Logging/logdata.h>
#include <Main/engine.h>
#include <Main/rl_subsystem_timers.h>
#include <Objects/object.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#if defined(__APPLE__) || defined(__linux__)
#define OGRL_BENCHMARK_BARRIER_SUPPORTED 1
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define OGRL_BENCHMARK_BARRIER_SUPPORTED 0
#endif

namespace RLBenchmark {
namespace {

using Clock = std::chrono::steady_clock;

// Six 120 Hz physics steps are one 20 Hz policy interval. Keeping the manual
// batch to one control interval amortizes outer-loop services without starving
// input/event handling for long simulation bursts.
const uint64_t kManualStepBatchSize = 6;

// Bound on how long a worker will wait at the readiness barrier for its peers.
// If this elapses, the run proceeds anyway (recorded as barrier_timed_out) --
// a single wedged/crashed peer must not hang the rest of a concurrency sweep.
const double kBarrierTimeoutSeconds = 60.0;
const double kBarrierPollSeconds = 0.002;

// Stage 4 in-process reset (Approach B), ported from exp-011. A cheap
// structural fingerprint of the scene -- object count/IDs/types/transforms/
// connections -- NOT a physics/combat equivalence proof (see Tools/rl/replay_compare.py
// and research-log OGRL-20260815-035 for the real one). This only catches
// "the reset left the world in a different shape than a fresh load," e.g. a
// leaked object or a missed re-registration.
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
    double measure_window_seconds = 0.0;
    std::string barrier_dir;
    int barrier_workers = 0;
    double progress_interval_seconds = 0.0;
    uint64_t steps_at_last_checkpoint = 0;
    Clock::time_point last_checkpoint_at;
    double engine_initialize_seconds = 0.0;
    double shader_preload_seconds = 0.0;
    double level_load_seconds = 0.0;
    double measurement_seconds = 0.0;
    double barrier_wait_seconds = 0.0;
    bool barrier_timed_out = false;
    bool barrier_applied = false;
    double reset_latency_seconds = 0.0;
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

#if OGRL_BENCHMARK_BARRIER_SUPPORTED
int CountReadyMarkers(const std::string& dir) {
    DIR* handle = opendir(dir.c_str());
    if (handle == nullptr) {
        return 0;
    }
    int count = 0;
    struct dirent* entry;
    while ((entry = readdir(handle)) != nullptr) {
        if (std::string(entry->d_name).rfind("ready.", 0) == 0) {
            ++count;
        }
    }
    closedir(handle);
    return count;
}
#endif

// Implements Stage 0.3's readiness barrier: write a marker file for this
// worker, then poll until `barrier_workers` markers are visible (or time out).
// This runs exactly once, right at the warmup -> measurement transition, so
// concurrently-launched workers all begin their timed measurement window at
// approximately the same wall-clock instant instead of drifting apart by
// however long their individual level loads happened to take.
void WaitForBarrier() {
    if (state.barrier_applied || state.barrier_workers <= 0 || state.barrier_dir.empty()) {
        state.barrier_applied = true;
        return;
    }
    state.barrier_applied = true;
#if OGRL_BENCHMARK_BARRIER_SUPPORTED
    mkdir(state.barrier_dir.c_str(), 0755);  // ignore EEXIST; harness may pre-create it too
    const std::string marker_path = state.barrier_dir + "/ready." + std::to_string(static_cast<long>(getpid()));
    FILE* marker = std::fopen(marker_path.c_str(), "w");
    if (marker != nullptr) {
        std::fclose(marker);
    }

    const Clock::time_point wait_started = Clock::now();
    while (CountReadyMarkers(state.barrier_dir) < state.barrier_workers) {
        if (SecondsSince(wait_started) > kBarrierTimeoutSeconds) {
            state.barrier_timed_out = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::duration<double>(kBarrierPollSeconds));
    }
    state.barrier_wait_seconds = SecondsSince(wait_started);
#else
    // No POSIX barrier support on this platform; the feature is a no-op and
    // measurement proceeds unsynchronized, same as before Stage 0.3.
#endif
}

}  // namespace

void Configure(bool enabled, uint64_t warmup_steps, uint64_t measure_steps, unsigned int seed,
               double measure_window_seconds, const std::string& barrier_dir, int barrier_workers,
               double progress_interval_seconds, bool reset_after_warmup) {
    state = State();
    state.enabled = enabled;
    state.warmup_steps = warmup_steps;
    state.measure_steps = measure_steps;
    state.seed = seed;
    state.measure_window_seconds = measure_window_seconds;
    state.barrier_dir = barrier_dir;
    state.barrier_workers = barrier_workers;
    state.progress_interval_seconds = progress_interval_seconds;
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

    if (state.measuring && state.measure_window_seconds > 0.0) {
        // Time-boxed measurement: still step in small batches so
        // OnTimestepComplete gets called often enough to notice the window
        // has elapsed, but do not assume a specific remaining step count.
        return static_cast<int>(kManualStepBatchSize);
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
        WaitForBarrier();
        state.measuring = true;
        state.measurement_started_at = Clock::now();
        state.last_checkpoint_at = state.measurement_started_at;
        RLSubsystemTimers::ResetAccumulators();  // exclude warmup and barrier wait from the breakdown
    }

    if (state.measuring) {
        ++state.measured_steps;

        if (state.progress_interval_seconds > 0.0) {
            const double since_checkpoint = SecondsSince(state.last_checkpoint_at);
            if (since_checkpoint >= state.progress_interval_seconds) {
                const uint64_t bucket_steps = state.measured_steps - state.steps_at_last_checkpoint;
                const double bucket_rate = since_checkpoint > 0.0 ? static_cast<double>(bucket_steps) / since_checkpoint : 0.0;
                std::cout << std::fixed << std::setprecision(6)
                          << "RL_BENCHMARK_PROGRESS {"
                          << "\"elapsed_seconds\":" << SecondsSince(state.measurement_started_at) << ','
                          << "\"bucket_seconds\":" << since_checkpoint << ','
                          << "\"bucket_steps\":" << bucket_steps << ','
                          << "\"bucket_steps_per_second\":" << bucket_rate << ','
                          << "\"cumulative_measured_steps\":" << state.measured_steps
                          << "}" << std::endl;
                state.last_checkpoint_at = Clock::now();
                state.steps_at_last_checkpoint = state.measured_steps;
            }
        }

        const bool hit_step_cap = state.measured_steps >= state.measure_steps;
        const bool hit_time_window =
            state.measure_window_seconds > 0.0 && SecondsSince(state.measurement_started_at) >= state.measure_window_seconds;
        if (hit_step_cap || hit_time_window) {
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
    // Emit to BOTH stdout and the engine log. On Windows the executable has no
    // attached console, so std::cout is discarded while the logger still writes
    // <write-dir>/logfile.txt -- which made every Windows benchmark look like it
    // had failed when the run had actually completed normally (2026-09-05). The
    // harness reads whichever it finds.
    std::ostringstream rl_out;
    rl_out << std::fixed << std::setprecision(6)
              << "RL_BENCHMARK_RESULT {"
              << "\"benchmark_completed\":" << (state.completed ? 1 : 0) << ','
              << "\"seed\":" << state.seed << ','
              << "\"fixed_physics_hz\":120,"
              << "\"warmup_steps\":" << state.warmup_steps << ','
              << "\"measured_steps\":" << state.measured_steps << ','
              << "\"steps_per_second\":" << steps_per_second << ','
              << "\"measurement_seconds\":" << state.measurement_seconds << ','
              << "\"measure_window_seconds\":" << state.measure_window_seconds << ','
              << "\"barrier_wait_seconds\":" << state.barrier_wait_seconds << ','
              << "\"barrier_timed_out\":" << (state.barrier_timed_out ? 1 : 0) << ','
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
              << "\"shader_preload_seconds\":" << state.shader_preload_seconds;
    if (RLSubsystemTimers::Enabled()) {
        rl_out << ',' << RLSubsystemTimers::ReportFragment(state.measurement_seconds);
    }
    rl_out << "}";
    const std::string rl_line = rl_out.str();
    std::cout << rl_line << std::endl;
    std::cout.flush();
    LOGI << rl_line << std::endl;
}

}  // namespace RLBenchmark
