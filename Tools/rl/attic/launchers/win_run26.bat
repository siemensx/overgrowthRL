@echo off
REM OGRL-20260924-007 run26_win: win-centric objective on the fixed learner.
REM   * resume the fixed-learner Phase-1b policy (run23_sel0, 262.97M) ONCE with
REM     --reset-critic + critic-only warm-up (actor frozen = a free same-config baseline);
REM   * gamma 0.99 -> 0.997, lambda 0.95 -> 0.975 (credit horizon ~0.6 s -> ~1.3 s GAE,
REM     ~11 s value horizon); reward profile "win" (KO 4, clear 12, self-KO 12);
REM   * rl_target_select 2 (nearest, native-AI rule) instead of the frozen-camera rule;
REM   * Windows throughput recipe from OGRL-20260924-001 arm A (n20/k4, n_steps 512,
REM     1 epoch, minibatch 128, update threads 1, engine above + 0xFFF, reset 20).
REM Supervisor: restarts on crash from run26's OWN checkpoint (no critic reset again).
REM Stop cleanly: write {"command":"stop"} to Tools\rl\runs\run26_win\control.json.
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] run26: another supervisor holds the lock - exiting >> C:\ogrl\run26.log
  exit /b 1
)
exit /b 0

:main
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL_clean
set RUN=run26_win
set BASE=%REPO%\Tools\rl\ppo\checkpoints\run23_sel0.pt
set CKPT=%REPO%\Tools\rl\ppo\checkpoints\%RUN%.pt
set OGRL_BINARY=C:\ogrl\optimization\BuildWinDet232Fixed_20260923\Release\Overgrowth.exe
set OGRL_ENGINE_PRIORITY=above
set OGRL_ENGINE_AFFINITY=0xFFF
set OGRL_LAUNCH_WAVE_SIZE=1
set OGRL_ALLOW_NENVS_CHANGE=1
cd /d %REPO%
set /a TRIES=0
:loop
set /a TRIES+=1
if exist "%CKPT%" (
  set START=--resume-from %CKPT% --reset-running-return
) else (
  set START=--resume-from %BASE% --reset-critic --value-warmup-updates 70
)
echo [%date% %time%] %RUN%: launch %TRIES% %START% >> C:\ogrl\run26.log
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml ^
  --shm-prefix /ogrl_r26_%RANDOM%%TRIES% --n-envs 20 --k-standby 4 --seed 26 --allow-n-envs-change ^
  --checkpoint-path %CKPT% %START% --run-id %RUN% ^
  --total-timesteps 400000000 --n-steps 512 --n-epochs 1 --minibatch-size 128 ^
  --gamma 0.997 --gae-lambda 0.975 --reward-profile win ^
  --entropy-coef 0.003 --entropy-coef-final 0.003 --entropy-anneal-steps 1000000 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 20 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 --d-min 1.0 ^
  --opponents-cap 3 --opp-keep-solo 0.0 --armed-stage 0 ^
  --collection-torch-threads 2 --update-torch-threads 1 --torch-interop-threads 1 ^
  --engine-config-line "rl_target_select: 2" ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 400 --periodic-eval-sampled 200 ^
  --no-tapes --no-native-capture --device cpu ^
  --purpose "OGRL-20260924-007 win-centric objective, gamma .997, nearest targeting, n20k4" >> C:\ogrl\%RUN%.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] %RUN%: exited %RC% >> C:\ogrl\run26.log
taskkill /F /IM Overgrowth.exe >nul 2>&1
if "%RC%"=="0" exit /b 0
if %TRIES% GEQ 40 exit /b 1
ping -n 31 127.0.0.1 >nul
goto loop
