@echo off
REM run21 supervisor -- restarts training forever, snapshots checkpoints, rolls logs.
REM Power: keep the machine awake for the life of this window.
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
powercfg /change monitor-timeout-ac 0 >nul 2>&1

set REPO=C:\ogrl\overgrowthRL
set CKPT=%REPO%\Tools\rl\ppo\checkpoints\run21_win.pt
set SNAP=%REPO%\Tools\rl\ppo\checkpoints\snapshots
set RUNDIR=%REPO%\Tools\rl\runs\run21_win
if not exist "%SNAP%" mkdir "%SNAP%"
cd /d %REPO%

:loop
REM --- snapshot the checkpoint (hourly-ish, keyed by timestamp) ---
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set TS=%%t
if exist "%CKPT%" copy /y "%CKPT%" "%SNAP%\run21_win_%TS%.pt" >nul 2>&1
REM keep only the newest 24 snapshots
powershell -NoProfile -Command "Get-ChildItem '%SNAP%\*.pt' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 24 | Remove-Item -Force" >nul 2>&1

REM --- roll the run logs before they can fill the disk ---
powershell -NoProfile -Command "foreach($f in 'metrics.jsonl','episodes.jsonl'){$p=Join-Path '%RUNDIR%' $f; if((Test-Path $p) -and ((Get-Item $p).Length -gt 150MB)){$t=Get-Content $p -Tail 40000; Set-Content $p $t}}" >nul 2>&1

if not exist "%CKPT%" (
  echo [%date% %time%] no checkpoint at %CKPT% - starting cold >> C:\ogrl\supervisor.log
  set RESUME=
) else (
  set RESUME=--resume-from %CKPT%
)
echo [%date% %time%] launching train_vec >> C:\ogrl\supervisor.log
set OGRL_ALLOW_NENVS_CHANGE=1
python -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml ^
  --shm-prefix /ogrl_w%RANDOM% --n-envs 10 --k-standby 2 --seed 21 ^
  --checkpoint-path %CKPT% %RESUME% --run-id run21_win ^
  --total-timesteps 4000000000 --n-steps 256 --n-epochs 1 --minibatch-size 128 ^
  --entropy-coef 0.003 --entropy-coef-final 0.003 --entropy-anneal-steps 1000000 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 50 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 ^
  --opponents-cap 3 --armed-stage 0 --gate-eval-episodes 30 ^
  --no-tapes --no-native-capture >> C:\ogrl\run21_win.log 2>&1

echo [%date% %time%] train_vec exited, restarting in 30s >> C:\ogrl\supervisor.log
taskkill /F /IM Overgrowth.exe >nul 2>&1
ping -n 31 127.0.0.1 >nul
goto loop
