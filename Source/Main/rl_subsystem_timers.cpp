#include "rl_subsystem_timers.h"

#include <chrono>
#include <sstream>

namespace RLSubsystemTimers {
namespace {

using Clock = std::chrono::steady_clock;

bool g_enabled = false;

struct Accumulator {
    double seconds = 0.0;
    uint64_t calls = 0;
};

Accumulator g_accumulators[kZoneCount];

}  // namespace

void SetEnabled(bool enabled) {
    g_enabled = enabled;
}

bool Enabled() {
    return g_enabled;
}

void ResetAccumulators() {
    for (auto& accumulator : g_accumulators) {
        accumulator = Accumulator();
    }
}

double SecondsFor(Zone zone) {
    return g_accumulators[zone].seconds;
}

uint64_t CallCountFor(Zone zone) {
    return g_accumulators[zone].calls;
}

std::string ReportFragment(double total_step_seconds) {
    // kZoneObjectUpdates wraps SceneGraph::Update's per-object dispatch loop,
    // which is where MovementObject::Update (character AngelScript logic) is
    // invoked; kZoneAnimation is a *nested* timer taken specifically around
    // RiggedObject::Update, which MovementObject::Update calls internally.
    // character_script_seconds is therefore derived, not independently
    // measured: object_updates_seconds - animation_seconds. It still includes
    // whatever non-character per-object update cost is in the same loop
    // (items, decals, hotspots, ...), which the Stage 0.6 native profile is
    // the authority for disambiguating -- this accumulator is a cheap,
    // always-available corroboration, not a replacement for it.
    const double level_script_seconds = SecondsFor(kZoneLevelScript);
    const double bullet_seconds = SecondsFor(kZoneBulletWorld);
    const double object_updates_seconds = SecondsFor(kZoneObjectUpdates);
    const double animation_seconds = SecondsFor(kZoneAnimation);
    const double obs_extraction_seconds = SecondsFor(kZoneObsExtraction);
    const double character_script_seconds = object_updates_seconds - animation_seconds;
    // Directly measured AngelScript character Update(), vs the derived figure above.
    // The script runs on a PERIOD: 1 for the controlled character, 4 for AI
    // characters (movementobject.cpp:1445), so calls != ticks.
    const double character_script_measured = SecondsFor(kZoneCharacterScript);
    const double accounted_seconds = level_script_seconds + bullet_seconds + object_updates_seconds + obs_extraction_seconds;
    const double other_seconds = total_step_seconds > accounted_seconds ? total_step_seconds - accounted_seconds : 0.0;

    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(6);
    out << "\"level_script_seconds\":" << level_script_seconds << ','
        << "\"level_script_calls\":" << CallCountFor(kZoneLevelScript) << ','
        << "\"bullet_seconds\":" << bullet_seconds << ','
        << "\"bullet_calls\":" << CallCountFor(kZoneBulletWorld) << ','
        << "\"object_updates_seconds\":" << object_updates_seconds << ','
        << "\"object_updates_calls\":" << CallCountFor(kZoneObjectUpdates) << ','
        << "\"animation_seconds\":" << animation_seconds << ','
        << "\"animation_calls\":" << CallCountFor(kZoneAnimation) << ','
        << "\"character_script_seconds_derived\":" << character_script_seconds << ','
        << "\"obs_extraction_seconds\":" << obs_extraction_seconds << ','
        << "\"obs_extraction_calls\":" << CallCountFor(kZoneObsExtraction) << ','
        << "\"other_seconds\":" << other_seconds;
    out << ",\"character_script_measured_seconds\":" << character_script_measured
        << ",\"character_script_calls\":" << CallCountFor(kZoneCharacterScript);
    return out.str();
}

ScopedZone::ScopedZone(Zone zone) : zone_(zone), active_(g_enabled) {
    if (active_) {
        start_ = Clock::now();
    }
}

ScopedZone::~ScopedZone() {
    if (active_) {
        Accumulator& accumulator = g_accumulators[zone_];
        accumulator.seconds += std::chrono::duration<double>(Clock::now() - start_).count();
        ++accumulator.calls;
    }
}

}  // namespace RLSubsystemTimers
