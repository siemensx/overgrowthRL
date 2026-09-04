#include "rng_streams.h"

#include <cmath>

namespace RngStreams {
namespace {

struct StreamState {
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    uint64_t draw_count = 0;
};

StreamState g_streams[static_cast<size_t>(Stream::kCount)];

StreamState& Get(Stream stream) {
    return g_streams[static_cast<size_t>(stream)];
}

// splitmix64 (Sebastiano Vigna, public domain).
uint64_t SplitMix64Next(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

void SeedEpisode(unsigned int seed) {
    for (auto& stream : g_streams) {
        // Mix the raw seed through one splitmix64 round per stream so
        // kGameplay and kCosmetic do not start from trivially related states
        // even when seeded from the same episode seed (kCosmetic gets a
        // distinct fixed offset so the two streams are independent, not just
        // differently-initialized copies of the same sequence).
        stream.state = static_cast<uint64_t>(seed);
        stream.draw_count = 0;
    }
    Get(Stream::kCosmetic).state ^= 0xD1B54A32D192ED03ULL;  // distinct stream identity
    // Discard the first draw of each stream: splitmix64's first output from a
    // freshly-assigned small seed is otherwise strongly correlated with the
    // seed's low bits.
    SplitMix64Next(Get(Stream::kGameplay).state);
    SplitMix64Next(Get(Stream::kCosmetic).state);
}

uint64_t NextUInt64(Stream stream) {
    StreamState& s = Get(stream);
    ++s.draw_count;
    return SplitMix64Next(s.state);
}

float RangedRandomFloat(Stream stream, float min_value, float max_value) {
    if (min_value == max_value) {
        return min_value;
    }
    // NextUInt64's top 24 bits give a uniform float in [0,1) with full float
    // mantissa precision, avoiding the modulo-and-divide bias patterns of the
    // libc-rand()-based RangedRandomFloat it replaces.
    const uint64_t bits = NextUInt64(stream);
    const float unit = static_cast<float>(bits >> 40) / static_cast<float>(1ULL << 24);
    return unit * (max_value - min_value) + min_value;
}

int RangedRandomInt(Stream stream, int min_value, int max_value) {
    if (max_value <= min_value) {
        return min_value;
    }
    const uint64_t span = static_cast<uint64_t>(max_value - min_value) + 1;
    return static_cast<int>(NextUInt64(stream) % span) + min_value;
}

uint64_t DrawCount(Stream stream) {
    return Get(stream).draw_count;
}

}  // namespace RngStreams
