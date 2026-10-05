@echo off
REM OGRL-20261004-024 run37: two-teacher distillation, then a gentler move school (method: badbunny
REM research-log 2026-10-04 / OGRL-20261004-MOVE-SCHOOL.md).
REM Why: run36 learned ground fighting (ground-only 1v1 at d=1.0: 13/20 greedy) but S1's 70%% ground diet
REM wrecked normal play (canonical train .68/.11/.03) and 9M steps of S2 did not bring it back (normal 1v3
REM 0.07-0.11; part of the cause: hidden ground-only fights were ~25%% of normal-looking fights).
REM Phase A: a fresh LayerNorm student copies TWO teachers on 24 maps (101-124; t_held_203 never used):
REM   normal fights + hidden ground fights <- run37_teacher_normal.pt (= run35 final, canonical .89/.60/.32,
REM   held .79/.40/.23); announced ground-only fights <- run37_teacher_ground.pt (= run36 end of S1).
REM   Mix Toolsl\move_school\distill37.json (30%% ground-only, up to 1v2, all announced), ground d at 1.0.
REM Phase B: run37 RL from the student with Toolsl\move_schoolun37.json: S2b 25%% ground (30%% of them
REM   hidden) -> S3b 12%% ground (half hidden). Everything else as run36. 24 engines = one per map.
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] run37: another supervisor holds the lock - exiting >> C:\ogrl\run37.log
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
set LEVELS=arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_103.xml,arenas/t_train_104.xml,arenas/t_train_105.xml,arenas/t_train_106.xml,arenas/t_train_107.xml,arenas/t_train_108.xml,arenas/t_train_109.xml,arenas/t_train_110.xml,arenas/t_train_111.xml,arenas/t_train_112.xml,arenas/t_train_113.xml,arenas/t_train_114.xml,arenas/t_train_115.xml,arenas/t_train_116.xml,arenas/t_train_117.xml,arenas/t_train_118.xml,arenas/t_train_119.xml,arenas/t_train_120.xml,arenas/t_train_121.xml,arenas/t_train_122.xml,arenas/t_train_123.xml,arenas/t_train_124.xml
set FLAGS=--engine-config-line "rl_target_select: 2" --engine-config-line "rl_button_edges: 1" --engine-config-line "rl_no_feint: 1" --engine-config-line "rl_obs_omniscient: 1" --engine-config-line "rl_stick_deadzone: 0.3" --engine-config-line "rl_stance_walk: 1"
cd /d %REPO%
set /a TRIES=0

:distill
if exist C:\ogrl\run37_distill.done goto train
set /a TRIES+=1
echo [%date% %time%] run37: distill launch %TRIES% >> C:\ogrl\run37.log
%PY% -u Tools\rl\ppo\distill_vec.py --teacher %CK%\run37_teacher_normal.pt --teacher-ground %CK%\run37_teacher_ground.pt --out %CK%\run37_student.pt ^
  --levels %LEVELS% --n-envs 20 --k-standby 4 --n-steps 512 --shm-prefix /ogrl_d37_%RANDOM%%TRIES% --seed 37 ^
  --move-school --move-school-stages Tools\rl\move_school\distill37.json --force-stage 0 --force-ground-cap 1.0 --out-stage 0 --out-ground-cap 1.0 ^
  --layer-norm --total-steps 6000000 --teacher-acts-iters 60 --grad-steps 40 --minibatch 2048 ^
  --max-wall-hours 4 %FLAGS% >> C:\ogrl\run37_distill.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] run37: distill exited %RC% >> C:\ogrl\run37.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine -like '*ogrl_d37_*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if "%RC%"=="0" (
  echo done> C:\ogrl\run37_distill.done
  goto train
)
if %TRIES% GEQ 50 exit /b 1
ping -n 31 127.0.0.1 >nul
goto distill

:train
set RUN=run37_two_teacher
set CKPT=%CK%\%RUN%.pt
set /a TRIES+=1
if exist "%CKPT%" (
  set START=--resume-from %CKPT%
) else (
  set START=--resume-from %CK%\run37_student.pt
)
echo [%date% %time%] %RUN%: launch %TRIES% %START% >> C:\ogrl\run37.log
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels %LEVELS% ^
  --shm-prefix /ogrl_r37_%RANDOM%%TRIES% --n-envs 20 --k-standby 4 --seed 37 --allow-n-envs-change ^
  --checkpoint-path %CKPT% %START% --run-id %RUN% ^
  --total-timesteps 2000000000 --n-steps 512 --n-epochs 2 --minibatch-size 1024 ^
  --kl-mode adaptive --kl-hard-factor 4 --critic-full-batch --lr-min 0.00001 --lr-max 0.0003 ^
  --gamma 0.997 --gae-lambda 0.975 --reward-profile win_v2 --stall-target-weight 0 ^
  --entropy-coef 0.003 --entropy-target 0.5 --entropy-coef-max 0.01 --move-school --move-school-stages Tools\rl\move_school\run37.json --button-floor attack=0.1 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 20 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 --d-min 1.0 ^
  --opponents-cap 3 --opp-keep-solo 0.2 --opp-sampling learnability --armed-stage 0 ^
  --collection-torch-threads 2 --update-torch-threads 2 --torch-interop-threads 1 ^
  %FLAGS% ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 200 --periodic-eval-sampled 0 --periodic-eval-parallel 2 ^
  --max-wall-hours 4 ^
  --no-tapes --no-native-capture --device cpu ^
  --purpose "OGRL-20261004-024 run37: two-teacher distillation (normal=run35, ground=run36 S1) on 24 maps, gentler move school" >> C:\ogrl\%RUN%.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] %RUN%: exited %RC% >> C:\ogrl\run37.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine -like '*ogrl_r37_*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if "%RC%"=="0" exit /b 0
if %TRIES% GEQ 250 exit /b 1
ping -n 31 127.0.0.1 >nul
goto train
