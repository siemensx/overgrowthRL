@echo off
REM OGRL-20260926-003 run28_notimeout: fix the objective that paid the policy NOT to fight.
REM
REM Under run27's "win" profile the 1200-step cap was a BENIGN truncation: no terminal
REM penalty and V(s') bootstrapped in. Measured over 62,557 run27 1v3 episodes:
REM   EV(fight to a decision) = -2.61   (P(win|fought) = 0.304)
REM   EV(run the clock out)   = -1.92   <-- better by +0.69, and zero wins
REM   break-even win rate     = 0.322   <-- policy sat at 0.304, just below it
REM Below break-even this is self-reinforcing, and it showed: timeout share 0.061 -> 0.107,
REM mean length 565 -> 653, held-out 1v1 0.840 -> 0.715 and 1v3 0.460 -> 0.315.
REM
REM Changes vs run27, deliberately TWO:
REM   1. --reward-profile win_notimeout: timeout is a terminal -14 loss, full time_cost
REM      restored, so clear (+24) > lose fighting (-14) > run the clock (-22).
REM   2. resume from the 260M baseline, NOT run27's 305M weights. run27's last 40M steps
REM      made the policy worse at everything including 1v1; 260M is the better starting
REM      point and is the benchmark the target is stated against.
REM --reset-critic + critic-only warm-up because the reward profile changed: the old
REM critic estimates returns under an objective that no longer exists (see --reset-critic
REM help). The warm-up keeps fresh-critic advantages from moving the resumed actor.
REM Everything else is run27's recipe verbatim, so the reward change is the variable.
REM
REM Stop cleanly: write {"command":"stop"} to Tools\rl\runs\run28_notimeout\control.json.
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] run28: another supervisor holds the lock - exiting >> C:\ogrl\run28.log
  exit /b 1
)
exit /b 0

:main
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL_clean
set RUN=run28_notimeout
REM read-only reference into the other checkout; nothing writes this path
set BASE=C:\ogrl\overgrowthRL\Tools\rl\ppo\checkpoints\run21_baseline_260m.pt
set CKPT=%REPO%\Tools\rl\ppo\checkpoints\%RUN%.pt
set OGRL_BINARY=C:\ogrl\optimization\BuildWinFix_20260924\Release\Overgrowth.exe
set OGRL_ENGINE_PRIORITY=above
set OGRL_ENGINE_AFFINITY=0xFFF
set OGRL_LAUNCH_WAVE_SIZE=1
set OGRL_ALLOW_NENVS_CHANGE=1
cd /d %REPO%
set /a TRIES=0
:loop
set /a TRIES+=1
REM first launch only: fresh critic + actor-frozen warm-up, because the reward changed.
REM a supervisor restart resumes THIS run's own checkpoint and must NOT reset again.
if exist "%CKPT%" (
  set START=--resume-from %CKPT%
) else (
  set START=--resume-from %BASE% --reset-critic --value-warmup-updates 40
)
echo [%date% %time%] %RUN%: launch %TRIES% %START% >> C:\ogrl\run28.log
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml ^
  --shm-prefix /ogrl_r28_%RANDOM%%TRIES% --n-envs 20 --k-standby 4 --seed 28 --allow-n-envs-change ^
  --checkpoint-path %CKPT% %START% --run-id %RUN% ^
  --total-timesteps 400000000 --n-steps 512 --n-epochs 1 --minibatch-size 128 ^
  --gamma 0.997 --gae-lambda 0.975 --reward-profile win_notimeout ^
  --entropy-coef 0.003 --entropy-coef-final 0.003 --entropy-anneal-steps 1000000 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 20 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 --d-min 1.0 ^
  --opponents-cap 3 --opp-keep-solo 0.0 --armed-stage 0 ^
  --collection-torch-threads 2 --update-torch-threads 1 --torch-interop-threads 1 ^
  --engine-config-line "rl_target_select: 2" --engine-config-line "rl_button_edges: 1" ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 400 --periodic-eval-sampled 200 ^
  --no-tapes --no-native-capture --device cpu ^
  --purpose "OGRL-20260926-003 timeout is a terminal loss; resumed from 260M baseline" >> C:\ogrl\%RUN%.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] %RUN%: exited %RC% >> C:\ogrl\run28.log
taskkill /F /IM Overgrowth.exe >nul 2>&1
if "%RC%"=="0" exit /b 0
if %TRIES% GEQ 40 exit /b 1
ping -n 31 127.0.0.1 >nul
goto loop