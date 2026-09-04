#!/usr/bin/env bash
# Stage 0.7: vmmap/footprint composition of a single steady-state six-character
# worker. Splits clean file-backed vs dirty anonymous vs derived structure so
# Stage 7's shared-asset-mmap question can be answered from evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${OGRL_BINARY:-${repo_root}/BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth}"
level="${OGRL_LEVEL:-arenas/oval_arena.xml}"
hold_seconds="${OGRL_HOLD_SECONDS:-8}"
out_dir="${1:?usage: memory_composition.sh <output-dir>}"
mkdir -p "${out_dir}"

write_dir="$(mktemp -d)"
trap 'rm -rf "${write_dir}"' EXIT  # write-dir caches are regenerable; see OGRL-20260815-034 disk-full incident

"${binary}" --write-dir "${write_dir}" --working-dir "${repo_root}" \
  --disable-rendering --no-dialogues --benchmark \
  --benchmark-warmup-steps 600 --benchmark-steps 2000000 \
  --benchmark-measure-seconds "$(( hold_seconds + 5 ))" \
  --benchmark-seed 20260815 --level "${level}" \
  --config $'global_time_scale_mult: 100\nskip_loading_pause: true\nhas_detected_settings: true' \
  > "${write_dir}/engine.log" 2>&1 &
engine_pid=$!

sleep 3  # clear warmup, reach steady state

vmmap --summary "${engine_pid}" > "${out_dir}/vmmap-summary.txt" 2>&1 || true
vmmap "${engine_pid}" > "${out_dir}/vmmap-full.txt" 2>&1 || true
footprint "${engine_pid}" > "${out_dir}/footprint.txt" 2>&1 || true
ps -o rss=,vsz= -p "${engine_pid}" > "${out_dir}/ps-rss-vsz.txt" 2>&1 || true

wait "${engine_pid}" || true
echo "memory composition captured in: ${out_dir}"
