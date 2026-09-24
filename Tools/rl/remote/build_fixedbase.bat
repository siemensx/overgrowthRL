@echo off
REM Build a fixed-base (RL_WINDOWS_FIXED_BASE=ON, AngelScript 2.32, no LTCG) Release
REM engine from a git checkout into its own build directory, at low priority so a
REM running training job keeps its cores. Usage: build_fixedbase.bat <repo> <builddir> <jobs>
set SRC=%~1
set BLD=%~2
set JOBS=%~3
if "%JOBS%"=="" set JOBS=4
REM The purchased asset tree is compiled in (AUX_DATA); without it the engine
REM dies at startup on Data/Sounds/voice/phonemes.txt.
if "%OGRL_AUX_DATA%"=="" set OGRL_AUX_DATA=C:/Program Files (x86)/Steam/steamapps/common/Overgrowth
echo [%date% %time%] configure %SRC% -> %BLD% > %BLD%.log
cmake -S %SRC%\Projects -B %BLD% -G "Visual Studio 17 2022" -A x64 -DRL_WINDOWS_FIXED_BASE=ON -DRL_MSVC_LTCG=OFF -DRL_ANGELSCRIPT_238=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 "-DAUX_DATA=%OGRL_AUX_DATA%" >> %BLD%.log 2>&1
if errorlevel 1 ( echo CONFIGURE_FAILED >> %BLD%.log & exit /b 1 )
echo [%date% %time%] build >> %BLD%.log
start "" /LOW /B /WAIT cmake --build %BLD% --config Release --target Overgrowth -- /m:%JOBS% /v:minimal >> %BLD%.log 2>&1
if errorlevel 1 ( echo BUILD_FAILED >> %BLD%.log & exit /b 1 )
echo [%date% %time%] BUILD_OK >> %BLD%.log
