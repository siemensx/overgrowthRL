@echo off
REM OGRL-20261002-005 run29_repair: the first run with the learner, curriculum and co-tenant fixes.
REM
REM What changed vs run28, and the measurement behind each change:
REM   * Learner. run28's KL guard stopped 99.7%% of updates at a median minibatch 12 of 80, so
REM     ~85%% of every rollout never produced a gradient for the actor OR the critic.
REM     Now: minibatch 1024, 2 epochs, --kl-mode adaptive (target_kl steers lr, hard stop at 4x),
REM     --critic-full-batch, lr starts at 1e-4.
REM   * Curriculum. Since 2026-09-09 every training episode was 1v3 at d=1.0 and the policy
REM     won 12-23%% of them in run28. Now 20%% 1v1, 40%% 1v2, 40%% 1v3 (--opp-keep-solo 0.2),
REM     still d=1.0, so the policy sees wins and keeps its 1v1 skill.
REM   * Reward win_v2: win_notimeout without per-step costs (time/stall/ragdoll), which made
REM     losing SOONER the better move once a fight looked lost.
REM   * Co-tenants. Engines + trainer at BelowNormal on CPUs 0-9 (0x3FF: one P-core + all
REM     E-cores). CPUs 10-13 (second P-core + both LP-E) stay free for the fbm Chrome/node
REM     workers. 12 engines instead of 24. Not elevated. --max-wall-hours 6 restarts the engine
REM     processes before their ~190 MB/h committed-memory leak can exhaust the pagefile
REM     (run28 crashed fbm Chrome and dwm twice that way, 09-27 and 09-28).
REM Critic reset + 20 critic-only warm-up updates on the FIRST launch only (reward changed).
REM
REM Stop cleanly: write {"command":"stop"} to Tools\rl\runs\run29_repair\control.json.
REM Pause for an fbm burst: {"command":"pause"} then {"command":null}.
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] run29: another supervisor holds the lock - exiting >> C:\ogrl\run29.log
  exit /b 1
)
exit /b 0

:main
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL_clean
set RUN=run29_repair
if "%OGRL_RUN29_BASE%"=="" (
  set BASE=C:\ogrl\overgrowthRL\Tools\rl\ppo\checkpoints\run21_baseline_260m.pt
) else (
  set BASE=%OGRL_RUN29_BASE%
)
set CKPT=%REPO%\Tools\rl\ppo\checkpoints\%RUN%.pt
set OGRL_BINARY=C:\ogrl\optimization\BuildWinFix_20260924\Release\Overgrowth.exe
set OGRL_ENGINE_PRIORITY=below
set OGRL_ENGINE_AFFINITY=0x3FF
set OGRL_TRAINER_PRIORITY=below
set OGRL_TRAINER_AFFINITY=0x3FF
set OGRL_LAUNCH_WAVE_SIZE=1
set OGRL_ALLOW_NENVS_CHANGE=1
cd /d %REPO%
set /a TRIES=0
:loop
set /a TRIES+=1
if exist "%CKPT%" (
  set START=--resume-from %CKPT%
) else (
  set START=--resume-from %BASE% --reset-critic --value-warmup-updates 20
)
echo [%date% %time%] %RUN%: launch %TRIES% %START% >> C:\ogrl\run29.log
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml ^
  --shm-prefix /ogrl_r29_%RANDOM%%TRIES% --n-envs 10 --k-standby 2 --seed 29 --allow-n-envs-change ^
  --checkpoint-path %CKPT% %START% --run-id %RUN% ^
  --total-timesteps 2000000000 --n-steps 1024 --n-epochs 2 --minibatch-size 1024 ^
  --kl-mode adaptive --kl-hard-factor 4 --critic-full-batch --lr-min 0.00001 --lr-max 0.0003 ^
  --gamma 0.997 --gae-lambda 0.975 --reward-profile win_v2 --stall-target-weight 0 ^
  --entropy-coef 0.003 --entropy-coef-final 0.003 --entropy-anneal-steps 1000000 ^
  --learning-rate 0.0001 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 20 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 --d-min 1.0 ^
  --opponents-cap 3 --opp-keep-solo 0.2 --armed-stage 0 ^
  --collection-torch-threads 2 --update-torch-threads 2 --torch-interop-threads 1 ^
  --engine-config-line "rl_target_select: 2" --engine-config-line "rl_button_edges: 1" ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 200 --periodic-eval-sampled 0 --periodic-eval-parallel 2 ^
  --max-wall-hours 6 ^
  --no-tapes --no-native-capture --device cpu ^
  --purpose "OGRL-20261002-005 repair: full-batch learner, mixed 1v1/1v2/1v3, win_v2, co-tenant fence" >> C:\ogrl\%RUN%.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] %RUN%: exited %RC% >> C:\ogrl\run29.log
REM Kill only THIS run's engines (command line carries its shm prefix), never by image name.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine -like '*ogrl_r29_*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if "%RC%"=="0" exit /b 0
if %TRIES% GEQ 200 exit /b 1
ping -n 31 127.0.0.1 >nul
goto loop
