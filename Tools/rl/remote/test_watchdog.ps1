# Real tests for watchdog.ps1's decision logic, against throwaway files --
# never the live lock/log. Run on the trainer:
#   ssh trainer "powershell -File C:\ogrl\overgrowthRL\Tools\rl\remote\test_watchdog.ps1"
#
# These test the DECISION and RECOVERY logic (lock-free detection, staleness
# detection, kill-matching, idempotence) in isolation. They do not replace the
# live incident-recovery test (killing the real supervisor and confirming
# Task Scheduler + watchdog bring it back with the checkpoint intact) --
# that one is run once, by hand, against the real box, because it is
# destructive to a live run by nature and not something to automate on a
# schedule.

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$watchdog = Join-Path $here 'watchdog.ps1'
$tmp = Join-Path $env:TEMP ("ogrl_watchdog_test_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null

$pass = 0
$fail = 0
function Check([bool]$cond, [string]$name) {
  if ($cond) { Write-Output "PASS: $name"; $script:pass++ }
  else       { Write-Output "FAIL: $name"; $script:fail++ }
}

# Fake OGRL_Supervisor task so Start-ScheduledTask has something to call
# without touching the real one. Runs a no-op.
$fakeTaskName = 'OGRL_Watchdog_Test_NoOp'
Unregister-ScheduledTask -TaskName $fakeTaskName -Confirm:$false -ErrorAction SilentlyContinue
$fakeAction = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c exit 0'
Register-ScheduledTask -TaskName $fakeTaskName -Action $fakeAction `
  -Trigger (New-ScheduledTaskTrigger -Once -At (Get-Date).AddYears(5)) `
  -Settings (New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew) | Out-Null

function Run-Watchdog([string]$lock, [string]$log, [string]$wlog, [int]$staleMin = 5) {
  & $watchdog -LockPath $lock -LogPath $log -WatchdogLog $wlog `
    -SupervisorTask $fakeTaskName -StaleMinutes $staleMin -HeartbeatEveryN 1 2>&1 | Out-Null
}

# --- Test 1: healthy state (lock held, log fresh) does not intervene ---
$lock1 = Join-Path $tmp 'lock1'
$log1  = Join-Path $tmp 'log1'
$wlog1 = Join-Path $tmp 'wlog1'
Set-Content $log1 'update=1'
$holder = [System.IO.File]::Open($lock1, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
try {
  Run-Watchdog $lock1 $log1 $wlog1
  $content = if (Test-Path $wlog1) { Get-Content $wlog1 -Raw } else { '' }
  Check ($content -notmatch 'UNHEALTHY') "healthy (lock held, log fresh) does not intervene"
  Check ($content -match 'ok -') "healthy state still writes a heartbeat"
} finally {
  $holder.Close()
}

# --- Test 2: lock free (simulates the actual 2026-09-11 incident: the
#     supervisor's own process exited) triggers recovery ---
$lock2 = Join-Path $tmp 'lock2'   # never opened -- free by construction
$log2  = Join-Path $tmp 'log2'
$wlog2 = Join-Path $tmp 'wlog2'
Set-Content $log2 'update=1'
Run-Watchdog $lock2 $log2 $wlog2
$content2 = Get-Content $wlog2 -Raw
Check ($content2 -match 'UNHEALTHY: lock file free') "free lock is detected and triggers recovery"
Check ($content2 -match 'requested Start-ScheduledTask') "free lock triggers a supervisor relaunch"

# --- Test 3: lock held but log stale (a hang, not an exit -- the case Task
#     Scheduler's own restart-on-failure cannot see) triggers recovery ---
$lock3 = Join-Path $tmp 'lock3'
$log3  = Join-Path $tmp 'log3'
$wlog3 = Join-Path $tmp 'wlog3'
Set-Content $log3 'update=1'
(Get-Item $log3).LastWriteTime = (Get-Date).AddMinutes(-30)
$holder3 = [System.IO.File]::Open($lock3, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
try {
  Run-Watchdog $lock3 $log3 $wlog3 -staleMin 5
  $content3 = Get-Content $wlog3 -Raw
  Check ($content3 -match 'UNHEALTHY: log stale') "held lock + stale log (hang) is detected and triggers recovery"
} finally {
  $holder3.Close()
}

# --- Test 4: process-matching kills the right things by command line, not
#     by name alone (so it never touches an unrelated cmd.exe/python.exe) ---
$decoyLock = Join-Path $tmp 'lock4'   # free -> forces the kill path
$decoyLog  = Join-Path $tmp 'log4'
$decoyWlog = Join-Path $tmp 'wlog4'
Set-Content $decoyLog 'update=1'
$decoy = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'ping -n 30 127.0.0.1 >nul' -PassThru -WindowStyle Hidden
Start-Sleep -Milliseconds 500
try {
  Run-Watchdog $decoyLock $decoyLog $decoyWlog
  Start-Sleep -Milliseconds 500
  Check (-not $decoy.HasExited) "an unrelated cmd.exe (no run_forever.bat in its command line) is left running"
} finally {
  if (-not $decoy.HasExited) { Stop-Process -Id $decoy.Id -Force -ErrorAction SilentlyContinue }
}

# --- Test 5: idempotence -- running twice back-to-back on an already-healthy
#     state does not double-log or error ---
$lock5 = Join-Path $tmp 'lock5'
$log5  = Join-Path $tmp 'log5'
$wlog5 = Join-Path $tmp 'wlog5'
Set-Content $log5 'update=1'
$holder5 = [System.IO.File]::Open($lock5, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
try {
  Run-Watchdog $lock5 $log5 $wlog5
  Run-Watchdog $lock5 $log5 $wlog5
  $lines5 = if (Test-Path $wlog5) { (Get-Content $wlog5) } else { @() }
  Check ($lines5.Count -le 2) "two consecutive healthy checks produce at most one heartbeat each, no error spam"
} finally {
  $holder5.Close()
}

Unregister-ScheduledTask -TaskName $fakeTaskName -Confirm:$false -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue

Write-Output "----"
Write-Output "watchdog self-tests: $pass passed, $fail failed"
if ($fail -gt 0) { exit 1 }
exit 0
