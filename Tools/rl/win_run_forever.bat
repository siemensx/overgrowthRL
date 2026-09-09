@echo off
REM run21 supervisor -- restarts training forever, snapshots checkpoints, rolls logs.
REM
REM SINGLE INSTANCE ONLY. Two supervisors are fatal: each one runs
REM "taskkill /F /IM Overgrowth.exe" between cycles, so supervisor B kills
REM supervisor A's engines mid-episode, A dies with ShmWaitTimeout, restarts,
REM and B kills it again -- an infinite mutual-destruction loop. That is
REM exactly what happened 2026-09-08 23:59 when win_autostart.bat fired a
REM second time and started a second run_forever. The lock below prevents it:
REM handle 9 is held open for the whole run, and Windows opens ">" with no
REM sharing, so a second instance cannot acquire it and exits immediately.
call :acquire
exit /b %ERRORLEVEL%

:acquire
2>nul ( 9>C:\ogrl\run_forever.lock ( call :main ) ) || (
  echo [%date% %time%] another run_forever already holds the lock - exiting >> C:\ogrl\supervisor.log
  exit /b 1
)
exit /b 0

:main
REM Power: keep the machine awake for the life of this window.
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
powercfg /change monitor-timeout-ac 0 >nul 2>&1

set PY=C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PY%" (
  echo [%date% %time%] FATAL: interpreter missing at %PY% >> C:\ogrl\supervisor.log
  exit /b 1
)
set FAST=0
set REPO=C:\ogrl\overgrowthRL
set CKPT=%REPO%\Tools\rl\ppo\checkpoints\run21_win.pt
set SNAP=%REPO%\Tools\rl\ppo\checkpoints\snapshots
set RUNDIR=%REPO%\Tools\rl\runs\run21_win
if not exist "%SNAP%" mkdir "%SNAP%"
cd /d %REPO%

:loop
REM --- snapshot the checkpoint (hourly-ish, keyed by timestamp) ---
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set TS=%%t
if exist "%CKPT%" copy /y "%CKPT%" "%SNAP%\run21_win_%TS%.pt" >nul 2>&1
REM keep only the newest 24 snapshots
powershell -NoProfile -Command "Get-ChildItem '%SNAP%\*.pt' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 24 | Remove-Item -Force" >nul 2>&1

REM --- roll the run logs before they can fill the disk ---
powershell -NoProfile -Command "foreach($f in 'metrics.jsonl','episodes.jsonl'){$p=Join-Path '%RUNDIR%' $f; if((Test-Path $p) -and ((Get-Item $p).Length -gt 150MB)){$t=Get-Content $p -Tail 40000; Set-Content $p $t}}" >nul 2>&1

if not exist "%CKPT%" (
  echo [%date% %time%] no checkpoint at %CKPT% - starting cold >> C:\ogrl\supervisor.log
  set RESUME=
) else (
  set RESUME=--resume-from %CKPT%
)
echo [%date% %time%] launching train_vec >> C:\ogrl\supervisor.log
for /f %%s in ('powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"') do set T0=%%s
set OGRL_ALLOW_NENVS_CHANGE=1
%PY% -u Tools\rl\ppo\train_vec.py ^
  --repo-root %REPO% ^
  --levels arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml ^
  --shm-prefix /ogrl_w%RANDOM% --n-envs 10 --k-standby 2 --seed 21 ^
  --checkpoint-path %CKPT% %RESUME% --run-id run21_win ^
  --total-timesteps 4000000000 --n-steps 256 --n-epochs 1 --minibatch-size 128 ^
  --entropy-coef 0.003 --entropy-coef-final 0.003 --entropy-anneal-steps 1000000 ^
  --learning-rate 0.0003 --target-kl 0.02 --max-episode-steps 1200 ^
  --frame-stack 4 --act-period 4 --soft-reset --hard-reset-every 50 ^
  --d-max-start 1.0 --d-max-cap 1.0 --d-step 0.1 ^
  --opponents-cap 3 --armed-stage 0 --gate-eval-episodes 30 ^
  --no-tapes --no-native-capture >> C:\ogrl\run21_win.log 2>&1

for /f %%s in ('powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"') do set T1=%%s
set /a RAN=%T1%-%T0%
REM Back off on fast failures. A launch that dies in seconds is a broken
REM environment, not a transient engine hang -- and the taskkill below would
REM otherwise fire every 30s forever, which is how one bad supervisor took the
REM whole run down on 2026-09-08.
if %RAN% LSS 120 (set /a FAST+=1) else (set FAST=0)
set WAIT=31
if %FAST% GEQ 3 set WAIT=121
if %FAST% GEQ 6 set WAIT=601
echo [%date% %time%] train_vec exited after %RAN%s (fastfail=%FAST%), restarting in %WAIT%s >> C:\ogrl\supervisor.log
taskkill /F /IM Overgrowth.exe >nul 2>&1
ping -n %WAIT% 127.0.0.1 >nul
goto loop
