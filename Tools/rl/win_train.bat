@echo off
REM OGRL-20261007-008: the ONE training launcher. The scheduled task OGRL_Train runs
REM   cmd.exe /c C:\ogrl\overgrowthRL_v6\Tools\rl\win_train.bat <profile>
REM and run_profile.py does the rest (resume / fork / fresh, 4-hour recycle, engine cleanup).
REM Switch profiles from the Mac with Tools/rl/remote/switch_profile.sh <profile>.
cd /d %~dp0\..\..
"C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe" -u Tools\rl\run_profile.py %* >> C:\ogrl\run_profile_supervisor.out 2>&1
exit /b %ERRORLEVEL%
