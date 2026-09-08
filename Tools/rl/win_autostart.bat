@echo off
REM Waits for the Overgrowth build to succeed, verifies it, then hands off to
REM run_forever.bat. Survives the ssh session that started it.
set LOG=C:\ogrl\autostart.log
set EXE=C:\ogrl\overgrowthRL\BuildWin64\Release\Overgrowth.exe
echo [%date% %time%] autostart waiting for build >> %LOG%

:wait
if not exist "C:\ogrl\build.log" goto sleep
findstr /C:"BUILD_EXIT=0" C:\ogrl\build.log >nul 2>&1
if %ERRORLEVEL%==0 goto built
findstr /C:"BUILD_EXIT=1" C:\ogrl\build.log >nul 2>&1
if %ERRORLEVEL%==0 (
  echo [%date% %time%] BUILD FAILED - not starting training >> %LOG%
  findstr /C:": error" C:\ogrl\build.log >> %LOG%
  exit /b 1
)
:sleep
ping -n 61 127.0.0.1 >nul
goto wait

:built
if not exist "%EXE%" (
  echo [%date% %time%] BUILD_EXIT=0 but exe missing - aborting >> %LOG%
  exit /b 1
)
echo [%date% %time%] build OK: %EXE% >> %LOG%

REM Header must agree with shm_env.py or the run hangs silently (64 vs 76).
cd /d C:\ogrl\overgrowthRL
for /f %%h in ('python -c "import sys;sys.path.insert(0,'Tools/rl');import shm_env;print(shm_env._HEADER_SIZE)"') do set HDR=%%h
echo [%date% %time%] python header size = %HDR% >> %LOG%
if not "%HDR%"=="76" (
  echo [%date% %time%] HEADER MISMATCH - aborting >> %LOG%
  exit /b 1
)

echo [%date% %time%] starting run_forever >> %LOG%
start "" C:\ogrl\run_forever.bat
exit /b 0
