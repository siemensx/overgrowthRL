@echo off
REM OGRL-20261002-011 run30_v2: the "v2 agent" restart recommended in research-artifacts/OGRL-20261002-REPORT.md.
REM FROM SCRATCH (the 260M-400M policies are specialists of the old turbo controls and a 58%%-blind view).
REM   * obs schema v6 (branch feat/obs-v6-privileged): sees every enemy (rl_obs_omniscient) plus each
REM     enemy's AI state (targets_me, ai_attacking, group wait, sub-goal, will_throw_counter, ...).
REM   * controls: corrected + no feint + stick dead zone (dodge/frontkick) + stance walk (backpedal).
REM   * learner (changed at ~6M, OGRL-20261002-015): minibatch 128, 1 epoch, lr 3e-4, KL stop at 0.02,
REM     critic full batch. The Mac from-scratch A/B (same seed and init) showed the mb1024/2-epoch/
REM     adaptive learner stalling (0.49, d_max 0.15 at 3.5M) while the old small-batch learner climbed
REM     (0.77, d_max 0.65). critic-full-batch keeps the critic learning when the KL stop fires later.
REM   * entropy 0.003 constant from ~10M (OGRL-20261002-017): the bonus uses the UNsquashed Gaussian's
REM     entropy, which grows without bound in log_std, so at 0.0066 the stick sigma climbed 1.4 -> 2.15
REM     (bang-bang stick; the dead zone that dodge/frontkick need is rarely reached).
REM   * reward win_v2; curriculum from d_max 0.15 and 1 opponent, gates up to d=1.0 and 3 opponents,
REM     20%% 1v1 kept once advanced.
REM   * co-tenant fence identical to run29 (BelowNormal, 0x3FF, 12 engines, 6 h recycle).
REM Engine: C:\ogrl\optimization\BuildWinV6_20261002 built from C:\ogrl\overgrowthRL_v6 (OGRL_Build_V6).
REM Do NOT run this from overgrowthRL_clean: the v5 tree's Python cannot read v6 observations.
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] run30: another supervisor holds the lock - exiting >> C:\ogrl\run30.log
  exit /b 1
)
exit /b 0

:main
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL_v6
set RUN=run30_v2
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
  set START=
)
echo [%date% %time%] %RUN%: launch %TRIES% %START% >> C:\ogrl\run30.log
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml ^
  --shm-prefix /ogrl_r30_%RANDOM%%TRIES% --n-envs 10 --k-standby 2 --seed 30 --allow-n-envs-change ^
  --checkpoint-path %CKPT% %START% --run-id %RUN% ^
  --total-timesteps 2000000000 --n-steps 1024 --n-epochs 1 --minibatch-size 128 ^
  --kl-mode stop --critic-full-batch ^
  --gamma 0.997 --gae-lambda 0.975 --reward-profile win_v2 --stall-target-weight 0 ^
  --entropy-coef 0.003 --entropy-coef-final 0.003 --entropy-anneal-steps 1000000 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 20 ^
  --d-max-start 0.15 --d-max-cap 1.0 --d-step 0.1 --d-min 0.0 ^
  --opponents-cap 3 --opp-keep-solo 0.2 --opp-sampling learnability --armed-stage 0 ^
  --collection-torch-threads 2 --update-torch-threads 2 --torch-interop-threads 1 ^
  --engine-config-line "rl_target_select: 2" --engine-config-line "rl_button_edges: 1" --engine-config-line "rl_no_feint: 1" --engine-config-line "rl_obs_omniscient: 1" --engine-config-line "rl_stick_deadzone: 0.3" --engine-config-line "rl_stance_walk: 1" ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 200 --periodic-eval-sampled 0 --periodic-eval-parallel 2 ^
  --max-wall-hours 6 ^
  --no-tapes --no-native-capture --device cpu ^
  --purpose "OGRL-20261002-011 v2 agent from scratch: obs v6 omniscient, corrected-nofeint+deadzone+stance, full-batch learner, win_v2" >> C:\ogrl\%RUN%.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] %RUN%: exited %RC% >> C:\ogrl\run30.log
REM Kill only THIS run's engines (command line carries its shm prefix), never by image name.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine -like '*ogrl_r30_*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if "%RC%"=="0" exit /b 0
if %TRIES% GEQ 200 exit /b 1
ping -n 31 127.0.0.1 >nul
goto loop
