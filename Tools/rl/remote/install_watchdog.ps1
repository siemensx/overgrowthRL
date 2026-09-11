# Install the self-healing pair for run21_win: run_forever.bat as a proper
# Scheduled Task (so Windows itself restarts it if the process exits), plus a
# watchdog task that polls every 5 minutes for the case Task Scheduler cannot
# see on its own -- a hang, not an exit. See watchdog.ps1's header for why.
#
# Idempotent: safe to re-run after editing either task's settings.
# Run elevated (as Administrator) on the trainer host itself, e.g. via
#   ssh trainer "powershell -File C:\ogrl\overgrowthRL\Tools\rl\remote\install_watchdog.ps1"
# using an account already in the local Administrators group -- registering a
# task to run as SYSTEM requires that.

$ErrorActionPreference = 'Stop'

$repo          = 'C:\ogrl\overgrowthRL'
$supervisorBat = 'C:\ogrl\run_forever.bat'
$watchdogPs1   = Join-Path $repo 'Tools\rl\remote\watchdog.ps1'

if (-not (Test-Path $supervisorBat)) { throw "missing $supervisorBat" }
if (-not (Test-Path $watchdogPs1))   { throw "missing $watchdogPs1" }

# Shared resilience settings: this pair exists specifically because a power
# event coincided with the whole supervisor tree vanishing on 2026-09-11, so
# nothing here may be allowed to stand down on battery.
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit ([TimeSpan]::Zero) `
  -RestartCount 999 `
  -RestartInterval (New-TimeSpan -Minutes 1)

# NOT SYSTEM, NOT S4U: Overgrowth.exe fatals ("Problem getting documents
# path") under both -- neither loads pavlov's HKCU profile far enough for
# SHGetFolderPath(Personal) to resolve, even though the registry value itself
# is a plain local path (C:\Users\pavlov\Documents, not OneDrive-redirected).
# Learned live on 2026-09-11 installing this task: SYSTEM fast-failed every
# worker in 2s, S4U (as pavlov, no stored password) failed identically.
# This host stays logged on to the pavlov desktop continuously (it is how the
# original 44-hour run was launched), so InteractiveToken -- run as pavlov's
# own already-open session -- is what actually has a resolvable Documents
# folder, and needs no stored password either.
$principal = New-ScheduledTaskPrincipal -UserId 'pavlov' -LogonType Interactive -RunLevel Highest

# --- OGRL_Supervisor: run_forever.bat, restarted by Task Scheduler itself ---
$supervisorAction = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$supervisorBat`""
# AtStartup covers a reboot once pavlov's autologon session comes up; AtLogOn
# for pavlov specifically covers the case startup fires before that session
# exists. Interactive tasks need one of these to actually find a session.
$supervisorTriggers = @(
  (New-ScheduledTaskTrigger -AtStartup),
  (New-ScheduledTaskTrigger -AtLogOn -User 'pavlov')
)

Unregister-ScheduledTask -TaskName 'OGRL_Supervisor' -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName 'OGRL_Supervisor' `
  -Action $supervisorAction -Trigger $supervisorTriggers `
  -Settings $settings -Principal $principal `
  -Description 'run21_win training supervisor (win_run_forever.bat). Restarted automatically by Task Scheduler if the process exits; see watchdog.ps1 for the hang case.' `
  | Out-Null
Write-Output "registered OGRL_Supervisor"

# --- OGRL_Watchdog: polls every 5 min for the case Task Scheduler can't see ---
$watchdogAction = New-ScheduledTaskAction -Execute 'powershell.exe' `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$watchdogPs1`""
$watchdogTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
  -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
$startupTrigger = New-ScheduledTaskTrigger -AtStartup
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User 'pavlov'
$watchdogSettings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
  -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 3)

Unregister-ScheduledTask -TaskName 'OGRL_Watchdog' -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName 'OGRL_Watchdog' `
  -Action $watchdogAction -Trigger @($watchdogTrigger, $startupTrigger, $logonTrigger) `
  -Settings $watchdogSettings -Principal $principal `
  -Description 'Every 5 min: recover run21_win if the supervisor lock is free or the training log has gone stale. See watchdog.ps1.' `
  | Out-Null
Write-Output "registered OGRL_Watchdog"

Get-ScheduledTask -TaskName 'OGRL_Supervisor','OGRL_Watchdog' | Select-Object TaskName, State
