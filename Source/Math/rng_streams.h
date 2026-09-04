// Stage 2: RNG stream separation.
//
// Source/Math/enginemath.cpp's RangedRandomFloat/RangedRandomInt, and every
// direct call to libc rand(), previously drew from ONE process-global C rand()
// stream shared by gameplay-relevant logic (AI decision timing, target-bone
// selection, damage rolls -- see movementobject.cpp:1439, Data/Scripts/aschar.as)
// and purely cosmetic effects (blood, particles, sound pitch variation).
// Removing or reordering a single draw anywhere shifted every subsequent draw,
// which is exactly the mechanism behind the Stage 1 finding (research-log
// OGRL-20260815-035) that identical seeds do not currently reproduce.
//
// This header provides two independent, self-contained (not libc-rand-based)
// deterministic streams:
//   kGameplay  - AI/combat/physics-adjacent randomness. Seeded per episode
//                from the episode seed. Everything AngelScript sees through
//                rand()/RangedRandomFloat() draws from this stream (Data/Scripts
//                gameplay logic is gameplay by definition).
//   kCosmetic  - blood, particles, sound variation, decals, and anything else
//                that cannot affect the physics/combat transition function.
//                This is also the DEFAULT for any native call site not
//                explicitly migrated -- per the plan, "keep the existing
//                global rand() as ... the cosmetic stream only" is the safe
//                default until each site is individually classified.
//
// Uses splitmix64 (Vigna, public domain): pure 64-bit integer arithmetic, no
// libc/libstdc++ RNG implementation, so behavior does not depend on compiler
// or platform RNG internals -- required for the arm64-vs-x86_64 comparator in
// Stage 1 to have a chance at bounded (not chaotic) cross-arch divergence.
#pragma once

#include <cstdint>

namespace RngStreams {

enum class Stream {
    kGameplay = 0,
    kCosmetic = 1,
    kCount,
};

// Reseeds both streams. Called once per episode (engine init / in-process
// reset) from the same seed value used for RLBenchmark/ResetRLTrainingScenario,
// so gameplay-stream draws are reproducible across a fresh process and an
// in-process reset with identical inputs -- the actual point of this stage.
void SeedEpisode(unsigned int seed);

uint64_t NextUInt64(Stream stream);
float RangedRandomFloat(Stream stream, float min_value, float max_value);
int RangedRandomInt(Stream stream, int min_value, int max_value);  // inclusive

// Per-episode draw counters, for the Stage 1 state digest (research-log
// OGRL-20260815-035's comparator): "add a per-episode draw counter per stream
// to the state digest ... far more sensitive than catching the resulting
// behavioral divergence."
uint64_t DrawCount(Stream stream);

}  // namespace RngStreams
