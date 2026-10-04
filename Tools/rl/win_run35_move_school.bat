@echo off
REM OGRL-20261004-011 run35_move_school: "move school" -- teach a second way to win (method:
REM research-artifacts/OGRL-20261004-MOVE-SCHOOL.md in the badbunny repo).
REM Start: run32_maps12's latest checkpoint (run31 lineage, corrected controls, 12 maps), copied to
REM run35_seed.pt. Identical to run32 except --move-school: a staged share of GROUND-ONLY fights (air
REM attacks dropped, rule visible to the policy), advancing on its own (curriculum.MOVE_SCHOOL_STAGES):
REM   S1 ground school  70%% ground-only (1v1 only)    >= 6M steps, then ground 1v1 >= 60%% or 20M
REM   S2 mixed          40%% ground-only (up to 1v2)   >= 10M steps, then ground 1v2 >= 40%% or 30M
REM   S3 free+refresher 15%% ground-only (up to 1v3)   permanent
REM Reward unchanged (win_v2). Stage state is in the checkpoint, so the 6-h recycles resume mid-stage.
REM A snapshot is written to checkpoints\snapshots at every stage change.
REM 04:40 throughput change (OGRL-20261004-013): CPUs 0-9 measured only ~55%% busy with 10+2 engines (lockstep
REM waits), so 20 active + 4 standby (= 2 per map) with n_steps 512 keeps the 10,240-decision batch; 4-h recycle
REM bounds the engine commit leak (~0.4 GB fresh + 0.19 GB/h each -> ~27 GB peak; commit was 26/91 GB).
REM 05:30 (OGRL-20261004-015): --button-floor attack=0.1. After 3.5M steps of S1 the policy still threw ZERO
REM ground attacks: p(attack | grounded, enemy < 2 m) = 1e-4, so ground attacks were never sampled. The floor
REM keeps attack at p >= 0.05 everywhere (training only; greedy play unchanged). S1's clock was restarted.
REM 06:55 (OGRL-20261004-016): the floor now applies only while GROUNDED (mid-jump it launched the leg cannon
REM early: training 1v3 wins fell ~0.22 -> ~0.05; greedy play was unaffected, 8/20).
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] run35: another supervisor holds the lock - exiting >> C:\ogrl\run35.log
  exit /b 1
)
exit /b 0

:main
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL_v6
set RUN=run35_move_school
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
  set START=--resume-from %REPO%\Tools\rl\ppo\checkpoints\run35_seed.pt
)
echo [%date% %time%] %RUN%: launch %TRIES% %START% >> C:\ogrl\run35.log
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_103.xml,arenas/t_train_104.xml,arenas/t_train_105.xml,arenas/t_train_106.xml,arenas/t_train_107.xml,arenas/t_train_108.xml,arenas/t_train_109.xml,arenas/t_train_110.xml,arenas/t_train_111.xml,arenas/t_train_112.xml ^
  --shm-prefix /ogrl_r35_%RANDOM%%TRIES% --n-envs 20 --k-standby 4 --seed 35 --allow-n-envs-change ^
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
  --engine-config-line "rl_target_select: 2" --engine-config-line "rl_button_edges: 1" --engine-config-line "rl_no_feint: 1" --engine-config-line "rl_obs_omniscient: 1" --engine-config-line "rl_stick_deadzone: 0.3" --engine-config-line "rl_stance_walk: 1" ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 200 --periodic-eval-sampled 0 --periodic-eval-parallel 2 ^
  --max-wall-hours 4 ^
  --no-tapes --no-native-capture --device cpu ^
  --purpose "OGRL-20261004-011 move school: staged ground-only fights from run32 weights" >> C:\ogrl\%RUN%.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] %RUN%: exited %RC% >> C:\ogrl\run35.log
REM Kill only THIS run's engines (command line carries its shm prefix), never by image name.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine -like '*ogrl_r35_*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if "%RC%"=="0" exit /b 0
if %TRIES% GEQ 200 exit /b 1
ping -n 31 127.0.0.1 >nul
goto loop
