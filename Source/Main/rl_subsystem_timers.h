// Opt-in, low-overhead per-subsystem timing accumulators (Stage 0.5).
//
// Disabled by default; enabled only via --benchmark-subsystem-timers so the
// normal benchmark path (and shipping game) pays nothing but a bool check per
// zone. When enabled, each RL_SUBSYSTEM_ZONE records wall-clock time and call
// count into a small fixed array -- no allocation, no locking, single thread.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace RLSubsystemTimers {

enum Zone {
    kZoneLevelScript = 0,
    kZoneBulletWorld,
    kZoneObjectUpdates,
    kZoneAnimation,
    kZoneObsExtraction,
    kZoneCharacterScript,   // the AngelScript character Update() call itself
    kZoneBloodSurface,      // visual-only blood drip simulation, measured before deciding to gate it
    kZoneCount,
};

void SetEnabled(bool enabled);
bool Enabled();
void ResetAccumulators();

double SecondsFor(Zone zone);
uint64_t CallCountFor(Zone zone);

// Comma-separated `"key":value` fragments (no braces) for splicing into the
// existing RL_BENCHMARK_RESULT JSON line, plus a derived other_seconds and
// character_script_seconds (= object_updates - animation, see .cpp comment).
std::string ReportFragment(double total_step_seconds);

// RAII scoped accumulator. When disabled, cost is one Enabled() branch.
class ScopedZone {
public:
    explicit ScopedZone(Zone zone);
    ~ScopedZone();
    ScopedZone(const ScopedZone&) = delete;
    ScopedZone& operator=(const ScopedZone&) = delete;

private:
    Zone zone_;
    bool active_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace RLSubsystemTimers

#define RL_SUBSYSTEM_ZONE(zone_enum) RLSubsystemTimers::ScopedZone _rl_subsystem_zone_##zone_enum(RLSubsystemTimers::zone_enum)
