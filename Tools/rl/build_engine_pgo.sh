#!/usr/bin/env bash
# Reproducible PGO+ThinLTO engine build (Stage 3h, research-log OGRL-20260816-003).
#
# Three-stage pipeline, fully scripted from a clean tree (per the Stage 0.1
# reproducibility bar -- this must not depend on a hand-run profile step):
#   1. Build an instrumented binary (-fprofile-generate).
#   2. Run it against a small, fixed, representative set of scenarios to
#      generate profile data, merge with llvm-profdata.
#   3. Rebuild with that profile (-fprofile-use) plus -flto=thin.
#
# Measured result (two independent 10-rep paired-A/B batches, six-character):
# +17.2%/+17.6% for -flto=thin alone over the plain -O3 baseline
# (Tools/rl/build_engine.sh's default), then a further +20.1%/+21.1% for
# PGO+LTO over LTO-alone -- roughly +40% cumulative. Gated on Stage 1: the
# PGO+LTO binary was bit-identical (hash_chain_match, 0.0 max deviation) to
# the LTO-only binary on the same seed in this run; LTO-vs-non-LTO itself
# shows small bounded floating-point drift (max deviation ~0.0007, episode
# outcome unchanged) accepted under the documented-numeric regime -- see the
# log entry for the full characterization before trusting either without
# re-reading it.
#
# Costs, relative to Tools/rl/build_engine.sh: ~3x the build time (three
# full/partial compiles instead of one; each pass is itself fast, ~70s, since
# this codebase links quickly) and requires AUX_DATA to be reachable at BUILD
# time, not just at benchmark time, since stage 2 runs the engine.
#
# OGRL_BUILD_DIR overrides the final output directory (default BuildArm64PGO/,
# kept distinct from build_engine.sh's BuildArm64/ so neither script clobbers
# the other's output). OGRL_AUX_DATA / OGRL_BUILD_JOBS as in build_engine.sh.
# PGO training scenarios/seeds/step counts are fixed constants below, not
# environment-configurable, so this script's output is reproducible from a
# clean tree with no hidden inputs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
asset_root="${OGRL_AUX_DATA:-/Users/pavlov/Library/Application Support/Steam/steamapps/common/Overgrowth/Overgrowth.app/Contents/MacOS}"
build_jobs="${OGRL_BUILD_JOBS:-10}"
final_dir="${OGRL_BUILD_DIR:-${repo_root}/BuildArm64PGO}"

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT
gen_dir="${work_dir}/gen"
use_dir="${work_dir}/use"
profile_raw_dir="${work_dir}/profraw"
profile_data="${work_dir}/merged.profdata"
mkdir -p "${profile_raw_dir}"

common_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_OSX_ARCHITECTURES=arm64
  -DBUILD_OGDA=OFF
  -DENABLE_STEAMWORKS=OFF
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DAUX_DATA="${asset_root}"
  -DRL_NATIVE_ARM64_TRAINING=ON
)

echo "OGRL_PGO_STAGE 1/3: instrumented build"
cmake -S "${repo_root}/Projects" -B "${gen_dir}" "${common_args[@]}" \
  -DCMAKE_CXX_FLAGS="-fprofile-generate=${profile_raw_dir}" \
  -DCMAKE_C_FLAGS="-fprofile-generate=${profile_raw_dir}" \
  -DCMAKE_EXE_LINKER_FLAGS="-fprofile-generate=${profile_raw_dir}"
cmake --build "${gen_dir}" --target Overgrowth -j "${build_jobs}"

echo "OGRL_PGO_STAGE 2/3: training runs"
gen_binary="${gen_dir}/Overgrowth.app/Contents/MacOS/Overgrowth"
# Fixed, small, representative set: the three stock scenarios this project's
# whole benchmark suite already treats as canonical (Tools/rl/benchmark_config_arm64.yaml),
# six-character run twice (it dominates the RL training workload) plus one
# pass each of duel/four-character for coverage breadth.
pgo_runs=(
  "t2/vs/heavyvsheavy.xml:20260815"
  "og_story/23_Rock_Arena_2v2.xml:20260816"
  "arenas/oval_arena.xml:20260817"
  "arenas/oval_arena.xml:20260818"
)
run_index=0
for entry in "${pgo_runs[@]}"; do
  run_index=$((run_index + 1))
  level="${entry%%:*}"
  seed="${entry##*:}"
  write_dir="$(mktemp -d)"
  LLVM_PROFILE_FILE="${profile_raw_dir}/run-${run_index}-%p.profraw" "${gen_binary}" \
    --write-dir "${write_dir}" --working-dir "${repo_root}" \
    --disable-rendering --no-dialogues --benchmark \
    --benchmark-warmup-steps 600 --benchmark-steps 8000 \
    --benchmark-seed "${seed}" --level "${level}" \
    --config $'global_time_scale_mult: 100\nskip_loading_pause: true\nhas_detected_settings: true' \
    > /dev/null 2>&1
  rm -rf "${write_dir}"
done

xcrun llvm-profdata merge -output="${profile_data}" "${profile_raw_dir}"/*.profraw

echo "OGRL_PGO_STAGE 3/3: PGO+LTO build"
cmake -S "${repo_root}/Projects" -B "${use_dir}" "${common_args[@]}" \
  -DCMAKE_CXX_FLAGS="-flto=thin -fprofile-use=${profile_data} -Wno-profile-instr-out-of-date -Wno-backend-plugin" \
  -DCMAKE_C_FLAGS="-flto=thin -fprofile-use=${profile_data} -Wno-profile-instr-out-of-date -Wno-backend-plugin" \
  -DCMAKE_EXE_LINKER_FLAGS="-flto=thin"
cmake --build "${use_dir}" --target Overgrowth -j "${build_jobs}"

rm -rf "${final_dir}"
mkdir -p "$(dirname "${final_dir}")"
cp -R "${use_dir}/Overgrowth.app" "$(dirname "${final_dir}")/$(basename "${final_dir}").app.tmp"
mkdir -p "${final_dir}"
mv "$(dirname "${final_dir}")/$(basename "${final_dir}").app.tmp" "${final_dir}/Overgrowth.app"

binary_path="${final_dir}/Overgrowth.app/Contents/MacOS/Overgrowth"
echo "OGRL_BUILD_RESULT arch=arm64 build_dir=${final_dir} binary=${binary_path} pgo=1 lto=thin"
