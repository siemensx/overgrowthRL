@echo off
REM OGRL-20261005-001 run38: from scratch with the non-saturating architecture.
REM Why: the run21 lineage's actor froze (~95%% saturated tanh by ~180-260M steps; OGRL-20261004-018) and
REM copies of it into fresh networks lose unseen-map skill (held 1v3 23 -> 4/5; OGRL-20261004-024b).
REM Network: LayerNorm before every trunk tanh + attention over fighters (Necto/Nexto-style; OGRL-20261004-025).
REM Curriculum: run30's (difficulty ramp from 0.15, opponents unlocked 1->2->3, learnability mix), 24 maps
REM t_train_101..124 (t_held_203 never trained on), 20 active + 4 standby = one engine per map.
REM Learner: run35/36 settings (mb1024 x2 adaptive KL, critic full batch, win_v2), entropy target 0.5 (coef
REM <= 0.01) and the grounded attack floor (attack=0.1) to keep ground attacks explored. No move school.
REM Judge ONLY on canonical suite v2 every ~50M steps; first gate at ~50M: 1v1 train >= .60.
call :acquire %*
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main %* ) ) || (
  echo [%date% %time%] run38: another supervisor holds the lock - exiting >> C:\ogrl\run38.log
  exit /b 1
)
exit /b 0

:main
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL_v6
set CK=%REPO%\Tools\rl\ppo\checkpoints
set RUN=run38_scratch_attn
set CKPT=%CK%\%RUN%.pt
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
  set START=--layer-norm --entity-attention
)
echo [%date% %time%] %RUN%: launch %TRIES% %START% >> C:\ogrl\run38.log
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_103.xml,arenas/t_train_104.xml,arenas/t_train_105.xml,arenas/t_train_106.xml,arenas/t_train_107.xml,arenas/t_train_108.xml,arenas/t_train_109.xml,arenas/t_train_110.xml,arenas/t_train_111.xml,arenas/t_train_112.xml,arenas/t_train_113.xml,arenas/t_train_114.xml,arenas/t_train_115.xml,arenas/t_train_116.xml,arenas/t_train_117.xml,arenas/t_train_118.xml,arenas/t_train_119.xml,arenas/t_train_120.xml,arenas/t_train_121.xml,arenas/t_train_122.xml,arenas/t_train_123.xml,arenas/t_train_124.xml ^
  --shm-prefix /ogrl_r38_%RANDOM%%TRIES% --n-envs 20 --k-standby 4 --seed 38 --allow-n-envs-change ^
  --checkpoint-path %CKPT% %START% --run-id %RUN% ^
  --total-timesteps 2000000000 --n-steps 512 --n-epochs 2 --minibatch-size 1024 ^
  --kl-mode adaptive --kl-hard-factor 4 --critic-full-batch --lr-min 0.00001 --lr-max 0.0003 ^
  --gamma 0.997 --gae-lambda 0.975 --reward-profile win_v2 --stall-target-weight 0 ^
  --entropy-coef 0.003 --entropy-target 0.5 --entropy-coef-max 0.01 --button-floor attack=0.1 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 20 ^
  --d-max-start 0.15 --d-max-cap 1.0 --d-step 0.1 --d-min 0.0 ^
  --opponents-cap 3 --opp-keep-solo 0.2 --opp-sampling learnability --armed-stage 0 ^
  --collection-torch-threads 2 --update-torch-threads 6 --torch-interop-threads 1 ^
  --engine-config-line "rl_target_select: 2" --engine-config-line "rl_button_edges: 1" --engine-config-line "rl_no_feint: 1" --engine-config-line "rl_obs_omniscient: 1" --engine-config-line "rl_stick_deadzone: 0.3" --engine-config-line "rl_stance_walk: 1" ^
  --periodic-eval-steps 5000000 --periodic-eval-episodes 200 --periodic-eval-sampled 0 --periodic-eval-parallel 2 ^
  --max-wall-hours 4 ^
  --no-tapes --no-native-capture --device cpu ^
  --purpose "OGRL-20261005-001 run38: from scratch, LayerNorm + attention over fighters, 24 maps, run30 curriculum" >> C:\ogrl\%RUN%.out 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] %RUN%: exited %RC% >> C:\ogrl\run38.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='Overgrowth.exe'\" | Where-Object { $_.CommandLine -like '*ogrl_r38_*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if "%RC%"=="0" exit /b 0
if %TRIES% GEQ 300 exit /b 1
ping -n 31 127.0.0.1 >nul
goto loop
