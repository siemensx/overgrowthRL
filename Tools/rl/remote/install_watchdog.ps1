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

# --- 2026-09-26 (OGRL-20260926-002) ---------------------------------------
# Parameterized. This script used to hardcode run21_win and
# C:\ogrl\run_forever.bat, so the watchdog it installed kept watching run21's
# log forever. By the time run27 hung, OGRL_Watchdog was disabled -- and
# re-enabling it as installed would have killed run27 immediately, because
# run21_win.log was 9,901 minutes stale. Every run-specific path is now a
# required parameter, and -WatchOnly installs just the watchdog against an
# already-running supervisor task (e.g. OGRL_Train_run27) without touching it.
# --------------------------------------------------------------------------
param(
  [Parameter(Mandatory = $true)][string]$RunId,          # e.g. run27_win
  [Parameter(Mandatory = $true)][string]$Repo,           # e.g. C:\ogrl\overgrowthRL_clean
  [string]$LogPath        = '',                          # default: C:\ogrl\<RunId>.out
  [string]$SupervisorTask = 'OGRL_Supervisor',           # the task the watchdog restarts
  [string]$SupervisorBat  = '',                          # omit (or -WatchOnly) to leave the supervisor task alone
  [string]$SupervisorBatName = '',                       # e.g. win_run27.bat -- lets recovery kill THIS run's cmd.exe only
  [switch]$WatchOnly,                                    # install/refresh ONLY OGRL_Watchdog
  [int]   $StaleMinutes   = 15
)

$ErrorActionPreference = 'Stop'

$watchdogPs1   = Join-Path $Repo 'Tools\rl\remote\watchdog.ps1'
if (-not (Test-Path $watchdogPs1))   { throw "missing $watchdogPs1" }
if (-not $LogPath) { $LogPath = "C:\ogrl\$RunId.out" }
$heartbeatPath = Join-Path $Repo "Tools\rl\runs\$RunId\heartbeat.json"
if (-not (Test-Path $LogPath)) { throw "missing $LogPath -- refusing to install a watchdog aimed at a path that does not exist" }
if (-not (Get-ScheduledTask -TaskName $SupervisorTask -ErrorAction SilentlyContinue)) {
  throw "no scheduled task named '$SupervisorTask' -- the watchdog would have nothing to restart"
}
Write-Output "watching run=$RunId log=$LogPath heartbeat=$heartbeatPath restart-task=$SupervisorTask stale=$StaleMinutes min"

$supervisorBat = $SupervisorBat
if (-not $WatchOnly) {
  if (-not $supervisorBat) { throw "pass -SupervisorBat, or -WatchOnly to skip re-registering the supervisor task" }
  if (-not (Test-Path $supervisorBat)) { throw "missing $supervisorBat" }
}

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

if ($WatchOnly) {
  Write-Output "-WatchOnly: leaving supervisor task '$SupervisorTask' untouched (it may be mid-run)"
} else {
  Unregister-ScheduledTask -TaskName $SupervisorTask -Confirm:$false -ErrorAction SilentlyContinue
  Register-ScheduledTask -TaskName $SupervisorTask `
    -Action $supervisorAction -Trigger $supervisorTriggers `
    -Settings $settings -Principal $principal `
    -Description "$RunId training supervisor ($supervisorBat). Restarted automatically by Task Scheduler if the process exits; see watchdog.ps1 for the hang case." `
    | Out-Null
  Write-Output "registered $SupervisorTask"
}

# --- OGRL_Watchdog: polls every 5 min for the case Task Scheduler can't see ---
# Every run-specific path is passed EXPLICITLY. watchdog.ps1 has no log
# default any more, precisely so a stale inherited one cannot kill a live run.
$watchdogAction = New-ScheduledTaskAction -Execute 'powershell.exe' `
  -Argument ("-NoProfile -ExecutionPolicy Bypass -File `"$watchdogPs1`"" +
             " -RunId `"$RunId`"" +
             " -LogPath `"$LogPath`" -HeartbeatPath `"$heartbeatPath`"" +
             " -SupervisorTask `"$SupervisorTask`" -StaleMinutes $StaleMinutes" +
             $(if ($SupervisorBatName) { " -SupervisorBatName `"$SupervisorBatName`"" } else { "" }))
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
  -Description "Every 5 min: recover $RunId if the supervisor lock is free or progress has gone stale for $StaleMinutes min. Heartbeat-aware: a run reporting disk_paused/control_paused is idle on purpose and is NOT killed. See watchdog.ps1." `
  | Out-Null
Write-Output "registered OGRL_Watchdog"

Get-ScheduledTask -TaskName $SupervisorTask,'OGRL_Watchdog' | Select-Object TaskName, State
