@echo off
REM Critic-detach causal test (review 2026-09-20): identical to run23_sel0 except
REM --critic-detach-shared. Waits for the running trainer to finish, then runs 2M
REM from 261M with benches at +1M and +2M (400 greedy x2 is done afterwards).
set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\ogrl\overgrowthRL
set LOG=C:\ogrl\detach.log
:wait
tasklist /FI "IMAGENAME eq python.exe" /FO CSV | findstr /I "python.exe" >nul
if %ERRORLEVEL%==0 ( ping -n 61 127.0.0.1 >nul & goto wait )
echo [%date% %time%] no trainer running - starting detach arm >> %LOG%
taskkill /F /IM Overgrowth.exe >nul 2>&1
cd /d %REPO%
set OGRL_ALLOW_NENVS_CHANGE=1
set OGRL_ENGINE_PRIORITY=normal
set OGRL_TRAINER_PRIORITY=above
set OGRL_ENGINE_AFFINITY=0xFFF
start "" /B powershell -NoProfile -Command "Start-Sleep 40; Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*train_vec*' } | ForEach-Object { (Get-Process -Id $_.ProcessId).PriorityClass = 'AboveNormal' }"
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml ^
  --shm-prefix /ogrl_dt%RANDOM% --n-envs 14 --k-standby 4 --seed 22 ^
  --checkpoint-path %REPO%\Tools\rl\ppo\checkpoints\run23_detach.pt --resume-from %REPO%\Tools\rl\ppo\checkpoints\run21_baseline_260m.pt --run-id run23_detach ^
  --total-timesteps 263284572 --n-steps 256 --n-epochs 1 --minibatch-size 128 ^
  --entropy-coef 0.003 --entropy-coef-final 0.003 --entropy-anneal-steps 1000000 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 50 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 --d-min 1.0 ^
  --opponents-cap 3 --opp-keep-solo 0.0 --armed-stage 0 --gate-eval-episodes 30 ^
  --engine-config-line "rl_target_select: 0" ^
  --periodic-eval-steps 1000000 --periodic-eval-episodes 400 --periodic-eval-sampled 200 ^
  --critic-detach-shared ^
  --no-tapes --no-native-capture >> C:\ogrl\run23_detach.log 2>&1
echo [%date% %time%] detach arm exited %ERRORLEVEL% >> %LOG%
taskkill /F /IM Overgrowth.exe >nul 2>&1
