@echo off
REM OGRL-20261004-019/020 run36: restore plasticity by distillation, then continue move school.
REM Phase A (distill_vec.py): a FRESH LayerNorm network copies run35's policy (teacher = frozen copy
REM   run36_teacher.pt) on the move-school scenario mix; teacher acts for the first 60 iterations, then the
REM   student (DAgger). 6M decisions, resumable across 4-h engine recycles (exit 75). Writes run36_student.pt,
REM   then C:\ogrl\run36_distill.done.
REM Phase B (train_vec.py): run36_plastic resumes from run36_student.pt with run35's exact RL settings
REM   (move school, grounded attack floor, ground-only difficulty ramp, 24 engines, entropy target). The
REM   student starts a fresh S1 window. Causal test of OGRL-20261004-018: does a non-saturated copy of the
REM   same policy learn ground fighting faster than run35 did (ground-only 1v1 wins 0.06-0.11, flat)?
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] run36: another supervisor holds the lock - exiting >> C:\ogrl\run36.log
  exit /b 1
)
exit /b 0

:main
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL_v6
set CK=%REPO%\Tools\rl\ppo\checkpoints
set OGRL_BINARY=C:\ogrl\optimization\BuildWinV6_20261002\Release\Overgrowth.exe
set OGRL_ENGINE_PRIORITY=below
set OGRL_ENGINE_AFFINITY=0x3FF
set OGRL_TRAINER_PRIORITY=below
set OGRL_TRAINER_AFFINITY=0x3FF
set OGRL_LAUNCH_WAVE_SIZE=1
set OGRL_ALLOW_NENVS_CHANGE=1
set LEVELS=arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_103.xml,arenas/t_train_104.xml,arenas/t_train_105.xml,arenas/t_train_106.xml,arenas/t_train_107.xml,arenas/t_train_108.xml,arenas/t_train_109.xml,arenas/t_train_110.xml,arenas/t_train_111.xml,arenas/t_train_112.xml
set FLAGS=--engine-config-line "rl_target_select: 2" --engine-config-line "rl_button_edges: 1" --engine-config-line "rl_no_feint: 1" --engine-config-line "rl_obs_omniscient: 1" --engine-config-line "rl_stick_deadzone: 0.3" --engine-config-line "rl_stance_walk: 1"
cd /d %REPO%
set /a TRIES=0

:distill
if exist C:\ogrl\run36_distill.done goto train
set /a TRIES+=1
echo [%date% %time%] run36: distill launch %TRIES% >> C:\ogrl\run36.log
%PY% -u Tools\rl\ppo\distill_vec.py --teacher %CK%\run36_teacher.pt --out %CK%\run36_student.pt ^
  --levels %LEVELS% --n-envs 20 --k-standby 4 --n-steps 512 --shm-prefix /ogrl_d36_%RANDOM%%TRIES% --seed 36 ^
  --move-school --layer-norm --total-steps 6000000 --teacher-acts-iters 60 --grad-steps 40 --minibatch 2048 ^
  --max-wall-hours 4 %FLAGS% >> C:\ogrl\run36_distill.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] run36: distill exited %RC% >> C:\ogrl\run36.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine -like '*ogrl_d36_*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if "%RC%"=="0" (
  echo done> C:\ogrl\run36_distill.done
  goto train
)
if %TRIES% GEQ 50 exit /b 1
ping -n 31 127.0.0.1 >nul
goto distill

:train
set RUN=run36_plastic
set CKPT=%CK%\%RUN%.pt
set /a TRIES+=1
if exist "%CKPT%" (
  set START=--resume-from %CKPT%
) else (
  set START=--resume-from %CK%\run36_student.pt
)
echo [%date% %time%] %RUN%: launch %TRIES% %START% >> C:\ogrl\run36.log
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels %LEVELS% ^
  --shm-prefix /ogrl_r36_%RANDOM%%TRIES% --n-envs 20 --k-standby 4 --seed 36 --allow-n-envs-change ^
  --checkpoint-path %CKPT% %START% --run-id %RUN% ^
  --total-timesteps 2000000000 --n-steps 512 --n-epochs 2 --minibatch-size 1024 ^
  --kl-mode adaptive --kl-hard-factor 4 --critic-full-batch --lr-min 0.00001 --lr-max 0.0003 ^
  --gamma 0.997 --gae-lambda 0.975 --reward-profile win_v2 --stall-target-weight 0 ^
  --entropy-coef 0.003 --entropy-target 0.5 --entropy-coef-max 0.01 --move-school --button-floor attack=0.1 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 20 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 --d-min 1.0 ^
  --opponents-cap 3 --opp-keep-solo 0.2 --opp-sampling learnability --armed-stage 0 ^
  --collection-torch-threads 2 --update-torch-threads 2 --torch-interop-threads 1 ^
  %FLAGS% ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 200 --periodic-eval-sampled 0 --periodic-eval-parallel 2 ^
  --max-wall-hours 4 ^
  --no-tapes --no-native-capture --device cpu ^
  --purpose "OGRL-20261004-020 run36: run35 distilled into a fresh LayerNorm net, then move school" >> C:\ogrl\%RUN%.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] %RUN%: exited %RC% >> C:\ogrl\run36.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine -like '*ogrl_r36_*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if "%RC%"=="0" exit /b 0
if %TRIES% GEQ 250 exit /b 1
ping -n 31 127.0.0.1 >nul
goto train
