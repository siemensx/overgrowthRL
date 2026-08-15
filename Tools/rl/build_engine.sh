#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${OGRL_BUILD_DIR:-${repo_root}/BuildRelease64}"
asset_root="${OGRL_AUX_DATA:-/Users/pavlov/Library/Application Support/Steam/steamapps/common/Overgrowth/Overgrowth.app/Contents/MacOS}"
build_jobs="${OGRL_BUILD_JOBS:-10}"

cmake -S "${repo_root}/Projects" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DBUILD_OGDA=OFF \
  -DENABLE_STEAMWORKS=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DFT_DISABLE_HARFBUZZ=TRUE \
  -DFT_DISABLE_BROTLI=TRUE \
  -DFT_DISABLE_PNG=TRUE \
  -DAUX_DATA="${asset_root}"

cmake --build "${build_dir}" --target Overgrowth -j "${build_jobs}"
