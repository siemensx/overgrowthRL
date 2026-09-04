#!/usr/bin/env bash
# Reproducible engine build for RL training/benchmarking.
#
# OGRL_ARCH selects the target architecture:
#   arm64   (default) - native Apple Silicon build, RL_NATIVE_ARM64_TRAINING=ON,
#                        AngelScript 2.38, no Steamworks/SDL2_net/OpenAL/Ogg/Vorbis.
#                        This is the build all current throughput baselines refer to
#                        (OGRL-20260815-029/-030) and it must be reachable by running
#                        this script alone, not by hand-configuring cmake in a worktree.
#   x86_64            - legacy Rosetta-path build, AngelScript 2.32, matches the
#                        pre-native baseline. Kept for A/B comparison against the
#                        native build and for the Stage 1 cross-architecture gate.
#
# OGRL_BUILD_DIR overrides the default per-arch build directory
# (BuildArm64/ or BuildRelease64/), OGRL_AUX_DATA overrides the asset root, and
# OGRL_BUILD_JOBS overrides the parallel build job count.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
arch="${OGRL_ARCH:-arm64}"
asset_root="${OGRL_AUX_DATA:-/Users/pavlov/Library/Application Support/Steam/steamapps/common/Overgrowth/Overgrowth.app/Contents/MacOS}"
build_jobs="${OGRL_BUILD_JOBS:-10}"

case "${arch}" in
  arm64)
    default_build_dir="${repo_root}/BuildArm64"
    ;;
  x86_64)
    default_build_dir="${repo_root}/BuildRelease64"
    ;;
  *)
    echo "error: unsupported OGRL_ARCH='${arch}' (expected 'arm64' or 'x86_64')" >&2
    exit 1
    ;;
esac
build_dir="${OGRL_BUILD_DIR:-${default_build_dir}}"

common_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_OSX_ARCHITECTURES="${arch}"
  -DBUILD_OGDA=OFF
  -DENABLE_STEAMWORKS=OFF
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DAUX_DATA="${asset_root}"
)

if [[ "${arch}" == "arm64" ]]; then
  # RL_NATIVE_ARM64_TRAINING forces ENABLE_STEAMWORKS=OFF itself and switches the
  # vendored AngelScript SDK to 2.38 (Projects/CMakeLists.txt:61-70,321-324). Do not
  # add the FT_DISABLE_* freetype flags here: the known-good hand-built arm64 tree
  # (OGRL-20260815-028/-030, worktree optimize-overgrowth-training-throughput-arm64-native)
  # was configured without them, and adding them is an untested config change.
  #
  # -flto=thin (research-log OGRL-20260816-002): +17.2%/+17.6% median throughput
  # across two independent 10-rep paired A/B batches on six-character, both well
  # outside the noise floor. Gated on Stage 1: strict-mode shows small divergence
  # (max deviation 0.00066 across all tracked quantities, episode outcome
  # identical) -- normal floating-point-reassociation drift from more aggressive
  # cross-TU inlining, not a logic bug; accepted under the documented-numeric
  # regime the same way a cross-architecture build would be. Link time for this
  # ~fully-clean build was 67s, not a meaningful cost.
  arch_args=(
    -DRL_NATIVE_ARM64_TRAINING=ON
    -DCMAKE_CXX_FLAGS=-flto=thin
    -DCMAKE_C_FLAGS=-flto=thin
    -DCMAKE_EXE_LINKER_FLAGS=-flto=thin
  )
else
  # Legacy x86_64/Rosetta path: matches the original (pre-Stage-0) build_engine.sh
  # exactly, including the freetype feature trims, so the pre-native baseline stays
  # reproducible for the Stage 1 cross-architecture comparator.
  arch_args=(
    -DFT_DISABLE_HARFBUZZ=TRUE
    -DFT_DISABLE_BROTLI=TRUE
    -DFT_DISABLE_PNG=TRUE
  )
fi

cmake -S "${repo_root}/Projects" -B "${build_dir}" "${common_args[@]}" "${arch_args[@]}"
cmake --build "${build_dir}" --target Overgrowth -j "${build_jobs}"

binary_path="${build_dir}/Overgrowth.app/Contents/MacOS/Overgrowth"
echo "OGRL_BUILD_RESULT arch=${arch} build_dir=${build_dir} binary=${binary_path}"
