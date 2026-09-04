// Stage 5.1/5.5 observation-extraction harness: runs RLObservation::Extract
// every `period` steps for the --rl-action-controller-id character, timed
// under RLSubsystemTimers::kZoneObsExtraction (Stage 0.5's accumulator, wired
// but previously unused since nothing called Extract), and optionally prints
// the first few results for eyeball verification. This is deliberately NOT
// the Stage 5.4 shm producer -- it is the measurement/verification harness
// that stage will replace, kept as an always-available diagnostic even after
// the real transport exists (obs_period ablation, Stage 6, reuses --rl-obs-period).
#pragma once

#include "rl_observation.h"

class Engine;

namespace RLObsTest {

// period <= 0 disables extraction entirely (default; zero measured cost).
// period == 1 extracts every step (the steady-state cost Stage 5.5 cares
// about); period == N extracts every Nth step, matching the act_period /
// obs_period concept Stage 6 will sweep. max_dumps caps how many of the
// extracted results get printed to stdout for inspection (0 = extract-only,
// no printing -- the mode used for cost measurement so I/O doesn't skew it).
// fov_profile lets --rl-obs-fov-npc-matched exercise Stage 5.2's alternative
// vision rule through this same harness rather than needing a separate one.
void Configure(int controller_id, int period, int max_dumps, RLObservation::FovProfile fov_profile = RLObservation::FovProfile::kOmnidirectional);
void Step(Engine* engine);

}  // namespace RLObsTest
