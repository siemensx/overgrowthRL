# Rendered fight recordings

Use `record_watch.py` for visual inspection of a checkpoint. It captures the
Overgrowth window, splits the command into one MP4 per completed fight, and
writes an episode summary. It is diagnostic footage, not canonical evaluation:
the assisted spectator camera can still affect legacy camera-dependent target
selection in the rendered engine path.

## Naming and preservation

Every recording command must use a new, descriptive `--out` stem. Include the
checkpoint step, scenario, seed, and a timestamp or another unique run ID.
For example:

```bash
record_id="run21_step463640342_t101_1v3u_seed900000_20260912T1900"
python3 Tools/rl/record_watch.py \
  --out "Tools/rl/runs/visual/${record_id}.mp4" \
  --screen 4 \
  -- \
  --checkpoint Tools/rl/ppo/checkpoints/run21_win_latest.pt \
  --level arenas/t_train_101.xml \
  --frame-stack 4 --act-period 4 \
  --opponents 3 --armed-count 0 --difficulty 1.0 \
  --seed 900000 --episodes 5 --device cpu \
  --spectator-fov 110
```

The recorder refuses to overwrite the master stem, episode MP4s, or the
`*_episodes.json` summary if any of them already exists. This is intentional:
do not reuse a stem after a failed or partial run. Pick another `record_id`.
`--overwrite` exists only for an explicitly deliberate replacement and should
not appear in commands copied for routine use.

The old files are never cleaned up by this tool. The temporary capture master
and episode-boundary event file are removed after successful splitting; the
per-fight MP4s and JSON summary remain.

## What the output means

For an output stem `X.mp4`, expect:

```text
X_ep00.mp4
X_ep01.mp4
...
X_episodes.json
```

The JSON records the checkpoint-independent episode seed, outcome, engine
steps, crop rectangle, and video time range. The normal capture is the
Overgrowth window only. `--keep-master` additionally keeps the full cropped
command movie at `X.mp4`; `--full-screen` is an explicit exception for display
debugging and is not acceptable as behavioral evidence.

Before interpreting a run, verify the crop and files:

```bash
jq '.window_rect, [.episodes[] | {episode, seed, outcome, steps, video_path}]' \
  "Tools/rl/runs/visual/${record_id}_episodes.json"
ffprobe -v error -show_entries format=duration:stream=width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  "Tools/rl/runs/visual/${record_id}_ep00.mp4"
```

The command prints `window crop: ...` before capture. On the Retina Mac, the
rectangle should be in physical display pixels and is normally about 2x the
logical window dimensions. If the first frame contains Codex, Terminal, or
System Settings, discard that run as failed visual evidence, keep its files,
and use a new stem after fixing the crop.

The default Mac display index is `4` (`Capture screen 0`), but it is machine
and display-layout dependent. Discover it with:

```bash
ffmpeg -f avfoundation -list_devices true -i ''
```

The terminal needs macOS Screen Recording permission. Keep `--spectator-fov`
between 1 and 179 degrees; 110 is the current readable 1v3 diagnostic value.
Do not use these camera settings to compare canonical win rates.
