#include "rl_obs_test.h"

#include "rl_observation.h"
#include "rl_subsystem_timers.h"

#include <Main/engine.h>
#include <Main/scenegraph.h>
#include <Objects/movementobject.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace RLObsTest {
namespace {

int g_controller_id = -1;
int g_period = 0;
int g_max_dumps = 0;
int g_dumps_printed = 0;
uint64_t g_step_counter = 0;
RLObservation::FovProfile g_fov_profile = RLObservation::FovProfile::kOmnidirectional;
// Reused across calls rather than a local in Step() -- a steady-state cost
// measurement should not include a per-step vector allocation that the real
// Stage 5.4 producer (a persistent shm-backed buffer) will never pay either.
std::vector<float> g_scratch_obs;

}  // namespace

void Configure(int controller_id, int period, int max_dumps, RLObservation::FovProfile fov_profile) {
    g_controller_id = controller_id;
    g_period = period;
    g_max_dumps = max_dumps;
    g_fov_profile = fov_profile;
    g_dumps_printed = 0;
    g_step_counter = 0;
    g_scratch_obs.clear();
}

void Step(Engine* engine) {
    if (g_controller_id < 0 || g_period <= 0) {
        return;
    }
    ++g_step_counter;
    if ((g_step_counter % static_cast<uint64_t>(g_period)) != 0) {
        return;
    }

    SceneGraph* scenegraph = engine->GetSceneGraph();
    if (scenegraph == nullptr) {
        return;
    }
    MovementObject* character = nullptr;
    for (Object* object : scenegraph->movement_objects_) {
        MovementObject* mo = static_cast<MovementObject*>(object);
        if (mo->controller_id == g_controller_id) {
            character = mo;
            break;
        }
    }
    if (character == nullptr) {
        return;
    }

    RLObservation::ObservationConfig config;
    config.fov_profile = g_fov_profile;
    bool truncated = false;
    int written = 0;
    {
        RL_SUBSYSTEM_ZONE(kZoneObsExtraction);
        written = RLObservation::Extract(engine, character, config, &g_scratch_obs, &truncated);
    }

    if (g_dumps_printed >= g_max_dumps) {
        return;
    }
    std::vector<const char*> names = RLObservation::FieldNames(config);
    std::printf("RL_OBS_DUMP step=%llu schema_version=%d los_rule_version=%d written=%d truncated=%d\n",
                static_cast<unsigned long long>(g_step_counter), RLObservation::kSchemaVersion,
                RLObservation::kLosRuleVersion, written, truncated ? 1 : 0);
    const int count = std::min(static_cast<int>(names.size()), written);
    for (int i = 0; i < count; ++i) {
        std::printf("RL_OBS_DUMP field[%d] %s=%f\n", i, names[i], g_scratch_obs[i]);
    }
    ++g_dumps_printed;
}

}  // namespace RLObsTest
