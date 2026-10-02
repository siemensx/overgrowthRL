#!/usr/bin/env bash
# Stage 0.6: capture a native arm64 profile (not Rosetta) of the six-character
# benchmark scenario, symbolicated, to replace/corroborate the x86_64 Rosetta
# sample (OGRL-20260815-005) that the plan says invalidates Stage 3's ordering
# until this exists.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${OGRL_BINARY:-${repo_root}/BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth}"
level="${OGRL_LEVEL:-arenas/oval_arena.xml}"
duration_seconds="${OGRL_SAMPLE_SECONDS:-12}"
out_path="${1:?usage: capture_native_profile.sh <output.sample.txt>}"

write_dir="$(mktemp -d)"
trap 'rm -rf "${write_dir}"' EXIT  # write-dir caches are regenerable; see OGRL-20260815-034 disk-full incident

"${binary}" --write-dir "${write_dir}" --working-dir "${repo_root}" \
  --disable-rendering --no-dialogues --benchmark \
  --benchmark-warmup-steps 600 --benchmark-steps 2000000 \
  --benchmark-measure-seconds "$(( duration_seconds + 5 ))" \
  --benchmark-seed 20260815 --level "${level}" \
  --config $'global_time_scale_mult: 100\nskip_loading_pause: true\nhas_detected_settings: true' \
  > "${write_dir}/engine.log" 2>&1 &
engine_pid=$!

# Let it clear warmup + settle into the steady-state measurement loop before sampling.
sleep 2

sample "${engine_pid}" "${duration_seconds}" -f "${out_path}" || true

wait "${engine_pid}" || true
echo "profile: ${out_path}"
