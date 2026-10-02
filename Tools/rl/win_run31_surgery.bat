@echo off
REM OGRL-20261002-020 run31_surgery: the "fast path" alternative to run30 (choose ONE; the trainer holds one run).
REM Starts from run27 (300.0M, best of the old approach: canonical suite 1v1 .89/.76, 1v2 .54/.35,
REM 1v3 .34/.05 train/held) transplanted EXACTLY into the v6 network by Tools/rl/ppo/surgery_v5_to_v6.py
REM (new inputs zero-weighted; mirrored laterals flipped; max |diff| 1.1e-5). Trains it under the v2 rules:
REM v6 omniscient observation, corrected-nofeint + deadzone + stance walk, mb1024/2-epoch adaptive learner,
REM critic full batch, win_v2, learnability opponent sampling at d=1.0, entropy 0.003, co-tenant fence.
REM First launch: --reset-critic + 20 critic-only warm-up updates (reward profile differs from run27's).
REM Checkpoint to copy first: C:\ogrl\overgrowthRL_v6\Tools\rl\ppo\checkpoints\run27_v6_surgery.pt
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] run31: another supervisor holds the lock - exiting >> C:\ogrl\run31.log
  exit /b 1
)
exit /b 0

:main
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL_v6
set RUN=run31_surgery
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
  set START=--resume-from %REPO%\Tools\rl\ppo\checkpoints\run27_v6_surgery.pt --reset-critic --value-warmup-updates 20
)
echo [%date% %time%] %RUN%: launch %TRIES% %START% >> C:\ogrl\run31.log
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml ^
  --shm-prefix /ogrl_r31_%RANDOM%%TRIES% --n-envs 10 --k-standby 2 --seed 31 --allow-n-envs-change ^
  --checkpoint-path %CKPT% %START% --run-id %RUN% ^
  --total-timesteps 2000000000 --n-steps 1024 --n-epochs 2 --minibatch-size 1024 ^
  --kl-mode adaptive --kl-hard-factor 4 --critic-full-batch --lr-min 0.00001 --lr-max 0.0003 ^
  --gamma 0.997 --gae-lambda 0.975 --reward-profile win_v2 --stall-target-weight 0 ^
  --entropy-coef 0.003 --entropy-coef-final 0.003 --entropy-anneal-steps 1000000 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 20 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 --d-min 1.0 ^
  --opponents-cap 3 --opp-keep-solo 0.2 --opp-sampling learnability --armed-stage 0 ^
  --collection-torch-threads 2 --update-torch-threads 2 --torch-interop-threads 1 ^
  --engine-config-line "rl_target_select: 2" --engine-config-line "rl_button_edges: 1" --engine-config-line "rl_no_feint: 1" --engine-config-line "rl_obs_omniscient: 1" --engine-config-line "rl_stick_deadzone: 0.3" --engine-config-line "rl_stance_walk: 1" ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 200 --periodic-eval-sampled 0 --periodic-eval-parallel 2 ^
  --max-wall-hours 6 ^
  --no-tapes --no-native-capture --device cpu ^
  --purpose "OGRL-20261002-020 run27 transplanted to v6, trained under v2 rules" >> C:\ogrl\%RUN%.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] %RUN%: exited %RC% >> C:\ogrl\run31.log
REM Kill only THIS run's engines (command line carries its shm prefix), never by image name.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine -like '*ogrl_r31_*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if "%RC%"=="0" exit /b 0
if %TRIES% GEQ 200 exit /b 1
ping -n 31 127.0.0.1 >nul
goto loop
