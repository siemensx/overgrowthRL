@echo off
REM OGRL-20261004-005 run32_maps12: run31_surgery's weights (378.4M; canonical v6-omni train .93/.65/.25,
REM held .77/.39/.14) continued under the same v2 rules with two changes, both aimed at what run31 showed:
REM   1. 12 training maps (t_train_101..112) instead of 3 -- the unseen-map gap is the largest left; 107-112
REM      were generated 2026-10-04 (gen_arena_map.py --randomize, seeds 107-112, 109/110/112 with clutter).
REM      t_held_203 stays out. 10 active + 2 standby engines = 12 = one map per engine (AGENTS invariant 6).
REM   2. --entropy-target 0.5: run31's fixed 0.003 bonus let entropy collapse 1.71 -> -1.14 with flat 1v3.
REM Gate (canonical suite v2, v6-omni): 1v3 held-out > 18/100 and train >= 29/100 by ~40M steps, else stop.
REM First launch seeds from run32_seed.pt (copy of run31_surgery_000378394332.pt) with 5 critic-only updates.
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] run32: another supervisor holds the lock - exiting >> C:\ogrl\run32.log
  exit /b 1
)
exit /b 0

:main
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL_v6
set RUN=run32_maps12
set CKPT=%REPO%\Tools\rl\ppo\checkpoints\%RUN%.pt
set OGRL_BINARY=C:\ogrl\optimization\BuildWinV6_20261002\Release\Overgrowth.exe
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
  set START=--resume-from %REPO%\Tools\rl\ppo\checkpoints\run32_seed.pt --value-warmup-updates 5
)
echo [%date% %time%] %RUN%: launch %TRIES% %START% >> C:\ogrl\run32.log
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_103.xml,arenas/t_train_104.xml,arenas/t_train_105.xml,arenas/t_train_106.xml,arenas/t_train_107.xml,arenas/t_train_108.xml,arenas/t_train_109.xml,arenas/t_train_110.xml,arenas/t_train_111.xml,arenas/t_train_112.xml ^
  --shm-prefix /ogrl_r32_%RANDOM%%TRIES% --n-envs 10 --k-standby 2 --seed 32 --allow-n-envs-change ^
  --checkpoint-path %CKPT% %START% --run-id %RUN% ^
  --total-timesteps 2000000000 --n-steps 1024 --n-epochs 2 --minibatch-size 1024 ^
  --kl-mode adaptive --kl-hard-factor 4 --critic-full-batch --lr-min 0.00001 --lr-max 0.0003 ^
  --gamma 0.997 --gae-lambda 0.975 --reward-profile win_v2 --stall-target-weight 0 ^
  --entropy-coef 0.003 --entropy-target 0.5 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 20 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 --d-min 1.0 ^
  --opponents-cap 3 --opp-keep-solo 0.2 --opp-sampling learnability --armed-stage 0 ^
  --collection-torch-threads 2 --update-torch-threads 2 --torch-interop-threads 1 ^
  --engine-config-line "rl_target_select: 2" --engine-config-line "rl_button_edges: 1" --engine-config-line "rl_no_feint: 1" --engine-config-line "rl_obs_omniscient: 1" --engine-config-line "rl_stick_deadzone: 0.3" --engine-config-line "rl_stance_walk: 1" ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 200 --periodic-eval-sampled 0 --periodic-eval-parallel 2 ^
  --max-wall-hours 6 ^
  --no-tapes --no-native-capture --device cpu ^
  --purpose "OGRL-20261004-005 run31 weights + 12 maps + entropy target 0.5" >> C:\ogrl\%RUN%.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] %RUN%: exited %RC% >> C:\ogrl\run32.log
REM Kill only THIS run's engines (command line carries its shm prefix), never by image name.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine -like '*ogrl_r32_*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if "%RC%"=="0" exit /b 0
if %TRIES% GEQ 200 exit /b 1
ping -n 31 127.0.0.1 >nul
goto loop
