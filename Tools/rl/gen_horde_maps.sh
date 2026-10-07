#!/usr/bin/env bash
# OGRL-20261007-003: the exact recipe of the horde maps (1v1..1v7 spawn groups), so they can be rebuilt
# after a Steam "verify files" or on a new host. Skips any map that already exists: gen_arena_map.py has
# no overwrite guard and a live map must never be regenerated under the same name.
# Seeds differ from the map number where the first seed failed validation (overlapping props / a padding
# pillar on a spawn) -- recorded here so the rebuild is byte-identical.
set -euo pipefail
cd "$(dirname "$0")"
DATA=$(python3 -c 'import paths; print(paths.data_dir())')
while read -r name seed extra; do
  [ -z "$name" ] && continue
  if [ -f "$DATA/Levels/arenas/$name.xml" ]; then echo "exists: $name"; continue; fi
  # shellcheck disable=SC2086
  python3 gen_arena_map.py --name "$name" --randomize --seed "$seed" --horde 7 $extra | grep -E "Wrote|REFUSING"
done <<'EOF'
t_horde_301 301
t_horde_302 302
t_horde_303 3031 --clutter 40
t_horde_304 304 --clutter 60
t_horde_305 3055 --clutter 50
t_horde_306 306
t_horde_307 307
t_horde_308 308 --clutter 40
t_horde_309 3091 --clutter 70
t_horde_310 310
t_horde_311 311 --clutter 50
t_horde_312 312 --clutter 30
t_horde_held_401 401 --clutter 40
EOF
