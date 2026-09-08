# Continuing run21 on the Windows trainer

Written 2026-09-08 for a fresh agent. The Mac (`Denyss-MacBook-Air`) has been
running `run21_mac`; the Windows box (`trainer-lan`, `DESKTOP-F1VOEDH`, 14
cores, 100 GB free) takes over so training can run indefinitely.

Read `Tools/rl/CURRICULUM.md` and `Tools/rl/EVALUATION_CONTRACT.md` first. They
are the contract; this file is only the mechanics.

---

## 0. State at handoff

* checkpoint `Tools/rl/ppo/checkpoints/run21_mac.pt`, **step ~281.7M**,
  curriculum `{d_max 1.0, opponents_max 3, armed_stage 6}` — the top rung
  (`C cats + mixed`: 3 armed hostiles, mixed weapons, cats in the pool).
* measured since 276M: **1v1 91.4%**, **1v2 85.4%**, **1v3 vs 3 armed 59.5%**.
* entropy ~4.5 (coef 0.012→0.008). Deliberately high: it was pinned at 0.003
  for 240M steps and the policy had collapsed to a jump-kick monoculture.

## 1. The two machines are NOT git-synced

`origin` is `github.com/siemensx/overgrowthRL` but the Windows clone is on a
different lineage (`245fe482`) and `git pull` there is a no-op. **Do not assume
git sync.** Files are copied directly:

```bash
for f in Source/Main/rl_shm_transport.cpp Data/Scripts/enemycontrol.as Data/Scripts/aschar.as \
         Tools/rl/curriculum.py Tools/rl/env.py Tools/rl/shm_env.py Tools/rl/ppo/train_vec.py \
         Tools/rl/ppo/watch.py Tools/rl/evaluate.py Tools/rl/gen_1v1_scenario.py; do
  scp -q "$f" "trainer-lan:C:/ogrl/overgrowthRL/$f"; done
```

Run PowerShell on the box with `bash Tools/rl/winps.sh trainer-lan < script.ps1`
(base64-encodes, so quoting survives). Plain `ssh trainer-lan "powershell -Command ..."`
mangles quotes — use the helper.

## 2. The engine MUST be rebuilt on Windows

The shm reset header grew 64 → **76 bytes** (added `reset_armed_count`,
`reset_weapon_type`, `reset_throw_aggression`). A stale `.exe` and the current
`shm_env.py` disagree and the run dies with an assertion or a silent hang.

```powershell
Set-Location C:\ogrl\overgrowthRL
python Tools\rl\gen_1v1_scenario.py          # regenerate the level script
cmake --build C:\ogrl\overgrowthRL\BuildWin64 --config Release --target Overgrowth -j 12
```

Verify: `python -c "import sys;sys.path.insert(0,'Tools/rl');import shm_env;print(shm_env._HEADER_SIZE)"` → **76**.

## 3. Assets that must match

Levels live in the Steam install, not the repo:
`C:\Program Files (x86)\Steam\steamapps\common\Overgrowth\Data\Levels\arenas\`

Training uses `t_train_101/102/104` only. **103/105/106 are regenerated but
un-gated — do not add them without running `validate_maps.py` first.** Copy from
the Mac if the box lacks them (its copies were once *different files* with the
same names, which invalidated months of eval numbers):

```bash
D=~/Library/Application\ Support/Steam/steamapps/common/Overgrowth/Overgrowth.app/Contents/MacOS/Data/Levels/arenas
T='C:/Program Files (x86)/Steam/steamapps/common/Overgrowth/Data/Levels/arenas'
for m in t_train_101 t_train_102 t_train_104 t_held_203; do scp -q "$D/$m.xml" "trainer-lan:$T/$m.xml"; done
```

Also clear stale navmesh caches after any map change — a baked navmesh is a
cache keyed by level *name*:
`Remove-Item 'C:/Program Files (x86)/Steam/.../Data/LevelNavmeshes/arenas/t_*' -Force`

## 4. Move the checkpoint over

```bash
scp Tools/rl/ppo/checkpoints/run21_mac.pt trainer-lan:C:/ogrl/overgrowthRL/Tools/rl/ppo/checkpoints/run21_win.pt
```

`_save_checkpoint` refuses to overwrite a checkpoint recording a HIGHER
`global_step` — so use a fresh name (`run21_win.pt`), or export
`OGRL_ALLOW_CHECKPOINT_REGRESSION=1` deliberately.

## 5. Start training

The Mac supervisor is bash. On Windows run `train_vec.py` directly, matching the
Mac's arguments (they are in `Tools/rl/supervise_run.sh`, the `launch()` body):

```powershell
Set-Location C:\ogrl\overgrowthRL
$env:OGRL_ALLOW_NENVS_CHANGE="1"
python -u Tools\rl\ppo\train_vec.py `
  --repo-root C:\ogrl\overgrowthRL `
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml `
  --shm-prefix /ogrl_w1 --n-envs 10 --k-standby 2 --seed 21 `
  --checkpoint-path Tools\rl\ppo\checkpoints\run21_win.pt --run-id run21_win `
  --total-timesteps 2000000000 --n-steps 256 --n-epochs 1 --minibatch-size 128 `
  --entropy-coef 0.012 --entropy-coef-final 0.008 --entropy-anneal-steps 8000000 `
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 `
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 50 `
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 `
  --opponents-cap 3 --armed-stage 6 `
  --no-tapes --no-native-capture 2>&1 | Tee-Object -FilePath C:\ogrl\run21_win.log
```

14 cores → try `--n-envs 10`. If sps is poor, drop to 8.

`--armed-stage 6` starts at the top rung, matching the checkpoint. The stage is
also restored from the checkpoint's `curriculum` dict, so this is belt-and-braces.

## 6. Confirm it is ACTUALLY running

Do not trust an exit code. Three checks:

```powershell
Get-Process Overgrowth | Measure-Object          # expect n_envs + k_standby
(Get-Content C:\ogrl\overgrowthRL\Tools\rl\runs\run21_win\metrics.jsonl -Tail 1 | ConvertFrom-Json).global_step
```

The step must **increase** between two reads a minute apart. Also confirm the
shm prefix on the live process matches what you launched — two "restarts" on
2026-09-07 silently did nothing because an old trainer survived `pkill` and the
new supervisor stood aside.

## 7. Traps that have already cost this project time

* **Disk.** `metrics.jsonl` reached 491 MB and `episodes.jsonl` 237 MB, filled
  the Mac's disk, and `--stop-below-free-gb` halted training for two hours.
  Windows has 100 GB, but roll them anyway — the Mac supervisor now does.
* **Never `pkill -f "Overgrowth"`** (or Windows `Stop-Process -Name Overgrowth`)
  while training runs: it matches the training workers. Probe scripts match
  `env-ogrl_w<pid>` instead.
* **Anchored string replaces must assert.** Two silent no-ops shipped in one
  day: a `watch.py --stage` flag that did nothing, and a level-script edit that
  never applied.
* **Compile-check AngelScript as the LAST action** after editing
  `Data/Scripts/*.as`. A variable declared inside a `switch` case took the run
  down for 8 minutes. Verify by stepping a live engine, not by eyeballing.
* **Headless never calls `Draw()`.** A level can train fine and segfault the
  moment a human opens it — use `Tools/rl/render_smoke.sh`.

## 8. Evaluation

Follow `EVALUATION_CONTRACT.md` exactly. The paired baseline is
`eval_snapshots/run21_mac_step247344226.pt` at 1v1 **0.925** / 1v2 0.741 / 1v3
0.562. Never judge progress from the training win rate — the curriculum sampler
holds it roughly constant by making the task harder; on 2026-09-07 it read
"plateau" while the paired eval showed +13.8 points.

## 9. Watching it fight

```powershell
python Tools\rl\ppo\watch.py --checkpoint Tools\rl\ppo\checkpoints\run21_win.pt `
  --level arenas/t_train_101.xml --stage 6 --difficulty 1.0 `
  --frame-stack 4 --act-period 4 --episodes 5 --max-episode-real-seconds 100
```

`--stage 0..6` replays any curriculum rung. Five episodes cannot distinguish a
broken policy from a normal one — at a 0.4 win rate, P(0 wins in 5) is 7.5%.
