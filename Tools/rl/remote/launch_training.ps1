# Launch a long training run on the Windows trainer under a Scheduled Task.
#
# Start-Process -WindowStyle Hidden with redirected output exits immediately and
# silently on this host (observed 2026-09-04), and Windows has no usable tmux, so
# a Scheduled Task with /RL HIGHEST plus a wrapper .bat that redirects its own
# output is the only pattern that survives SSH disconnects and logoff. A run
# measured in days must survive both.
#
# Parameters are filled by the caller; every value here is a decision recorded in
# the Stage C design (OGRL-20260904-060), not a default.
param(
  [int]    $NEnvs          = 8,
  [int]    $KStandby       = 3,
  [string] $RunId          = "run16",
  [int]    $Seed           = 1,
  [double] $EntropyCoef    = 0.012,
  [double] $EntropyFinal   = 0.003,
  [int]    $EntropyAnneal  = 8000000,
  [double] $StallWeight    = 0.02,
  [int]    $StallRamp      = 4000000,
  [long]   $TotalTimesteps = 120000000,
  [string] $ResumeFrom     = "C:\ogrl\overgrowthRL\Tools\rl\ppo\checkpoints\run15.pt",
  [string] $Levels         = "arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_103.xml,arenas/t_train_104.xml,arenas/t_train_105.xml,arenas/t_train_106.xml"
)

$py   = "C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe"
$repo = "C:\ogrl\overgrowthRL"
$log  = "C:\ogrl\$RunId.log"

$bat = @"
@echo off
cd /d $repo
"$py" "$repo\Tools\rl\ppo\train_vec.py" ^
  --n-envs $NEnvs --k-standby $KStandby ^
  --levels $Levels ^
  --resume-from "$ResumeFrom" ^
  --total-timesteps $TotalTimesteps ^
  --entropy-coef $EntropyCoef --entropy-coef-final $EntropyFinal --entropy-anneal-steps $EntropyAnneal ^
  --stall-target-weight $StallWeight --stall-ramp-steps $StallRamp ^
  --act-period 4 --frame-stack 4 --soft-reset --hard-reset-every 50 ^
  --device cpu --run-id $RunId --seed $Seed ^
  --no-tapes --no-native-capture ^
  --purpose "Stage C: entropy revival + stall tax + map axis, resumed from run15" ^
  --checkpoint-path "$repo\Tools\rl\ppo\checkpoints\$RunId.pt" ^
  > "$log" 2>&1
echo EXIT %ERRORLEVEL% >> "$log"
"@
Set-Content -Path "C:\ogrl\run_$RunId.bat" -Value $bat -Encoding ascii

schtasks /Delete /TN "OGRL_Train_$RunId" /F 2>&1 | Out-Null
schtasks /Create /TN "OGRL_Train_$RunId" /TR "C:\ogrl\run_$RunId.bat" /SC ONCE /ST 23:59 /RL HIGHEST /F 2>&1 | Select-Object -Last 1
schtasks /Run  /TN "OGRL_Train_$RunId" 2>&1 | Select-Object -Last 1
Start-Sleep -Seconds 45
"--- status ---"
schtasks /Query /TN "OGRL_Train_$RunId" /FO LIST 2>&1 | Select-String "Status|Last Result"
"engines running : " + ((Get-Process Overgrowth -ErrorAction SilentlyContinue | Measure-Object).Count)
"log             : " + $(if (Test-Path $log) { (Get-Item $log).Length.ToString() + " bytes" } else { "MISSING" })
if (Test-Path $log) { "--- log tail ---"; Get-Content $log | Select-Object -Last 8 | Out-String }
