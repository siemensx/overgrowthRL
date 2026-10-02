@echo off
REM Phase 1b of OPTIMIZATION_CONTRACT.md: the same actuator A/B on the FIXED learner
REM (raw-sample log-probs, exact-KL guard), old hyperparameters, bench 400 greedy + 200 sampled. Sequential arms, each resumed
REM from the 261M baseline into its own checkpoint, 10M steps, deterministic 200-episode
REM bench every 5M (periodic eval) -- the 10M bench IS the arm's result.
REM   ARMS: space-separated rl_target_select values, default "0 2" (control, nearest)
REM Single instance via the same lock as run_forever.bat.
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] phase1b: another supervisor holds the lock - exiting >> C:\ogrl\phase1.log
  exit /b 1
)
exit /b 0

:main
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
powercfg /change monitor-timeout-ac 0 >nul 2>&1
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL
set BASE=%REPO%\Tools\rl\ppo\checkpoints\run21_baseline_260m.pt
set LOG=C:\ogrl\phase1b.log
if "%ARMS%"=="" set ARMS=0 2
cd /d %REPO%
echo [%date% %time%] phase1b start arms=%ARMS% >> %LOG%
for %%A in (%ARMS%) do call :arm %%A
echo [%date% %time%] phase1b done >> %LOG%
exit /b 0

:arm
set SEL=%1
set RUN=run23_sel%SEL%
set CKPT=%REPO%\Tools\rl\ppo\checkpoints\%RUN%.pt
if exist "%CKPT%" (
  echo [%date% %time%] %RUN%: checkpoint exists, resuming it >> %LOG%
  set RESUME=--resume-from %CKPT%
) else (
  set RESUME=--resume-from %BASE%
)
echo [%date% %time%] %RUN%: launching, rl_target_select=%SEL% >> %LOG%
set OGRL_ALLOW_NENVS_CHANGE=1
REM scheduling (throughput_sweep 2026-09-20): engines normal, trainer above, engines off the LP E-cores -> best p10
set OGRL_ENGINE_PRIORITY=normal
set OGRL_TRAINER_PRIORITY=above
set OGRL_ENGINE_AFFINITY=0xFFF
start "" /B powershell -NoProfile -Command "Start-Sleep 40; Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*train_vec*' } | ForEach-Object { (Get-Process -Id $_.ProcessId).PriorityClass = 'AboveNormal' }"
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml ^
  --shm-prefix /ogrl_p2%SEL%%RANDOM% --n-envs 14 --k-standby 4 --seed 22 ^
  --checkpoint-path %CKPT% %RESUME% --run-id %RUN% ^
  --total-timesteps 271284572 --n-steps 256 --n-epochs 1 --minibatch-size 128 ^
  --entropy-coef 0.003 --entropy-coef-final 0.003 --entropy-anneal-steps 1000000 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 50 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 --d-min 1.0 ^
  --opponents-cap 3 --opp-keep-solo 0.0 --armed-stage 0 --gate-eval-episodes 30 ^
  --engine-config-line "rl_target_select: %SEL%" ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 400 --periodic-eval-sampled 200 ^
  --no-tapes --no-native-capture >> C:\ogrl\%RUN%.log 2>&1
echo [%date% %time%] %RUN%: exited %ERRORLEVEL% >> %LOG%
taskkill /F /IM Overgrowth.exe >nul 2>&1
ping -n 16 127.0.0.1 >nul
exit /b 0
