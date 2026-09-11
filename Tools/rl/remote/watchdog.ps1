# Self-healing check for the run21_win supervisor.
#
# Written 2026-09-11 after the supervisor's own process tree vanished at
# 01:30:00 (the same second Kernel-Power logged a power-source change,
# 100.118.2.91 -> unplugged) and nothing brought it back for the rest of the
# night: the lock file C:\ogrl\run_forever.lock was found FREE 17+ minutes
# later, meaning the cmd.exe running run_forever.bat had exited outright, not
# hung -- the 12 Overgrowth.exe workers it had spawned were left as orphans,
# alive but idle (near-zero CPU growth), waiting on an shm handshake with a
# python process that no longer existed. No component was watching the
# supervisor itself; the batch file only restarts train_vec.py when train_vec
# exits, and nothing restarts the batch file. The exact trigger was never
# pinned down (System/Application event logs show only the power-source-change
# event and four stale WER queue-flush entries pointing at July dumps) -- so
# this does not try to fix a specific cause, it detects "not making progress"
# by two independent signals and recovers either way. That is the same
# "treat hung as the crash signal" rule DEAD_ENDS.md already established for
# the engine's shm handshake, applied one level up, to the supervisor itself.
#
# Two independent unhealthy signals, either one triggers recovery:
#   1. The lock file is free (2>nul (9>lock) pattern) -- the supervisor's own
#      cmd.exe process has exited. Task Scheduler's own restart-on-failure
#      (OGRL_Supervisor task) should already have caught this within a
#      minute; this is the backstop for whatever timing window it misses.
#   2. The training log has not been written to in $StaleMinutes -- covers a
#      supervisor (or train_vec, or the engines) that is HUNG rather than
#      exited, which Task Scheduler cannot see: it only knows a process is
#      still running, not that the process is making progress.
#
# Idempotent and safe to run every 5 minutes forever: when healthy it writes
# nothing beyond a periodic heartbeat line.

param(
  [string]$LockPath      = 'C:\ogrl\run_forever.lock',
  [string]$LogPath       = 'C:\ogrl\run21_win.log',
  [string]$WatchdogLog   = 'C:\ogrl\watchdog.log',
  [string]$SupervisorTask = 'OGRL_Supervisor',
  [int]   $StaleMinutes  = 5,
  [int]   $HeartbeatEveryN = 12   # ~once/hour at a 5-min trigger interval
)

function Write-Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
  Add-Content -Path $WatchdogLog -Value $line
}

function Test-LockFree([string]$path) {
  # Mirrors the cmd.exe "2>nul ( 9>lock ( ... ) )" pattern: if we can open the
  # file with NO sharing allowed, nobody else holds it open -- the supervisor
  # that would hold it is not running.
  if (-not (Test-Path $path)) { return $true }
  try {
    $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    $fs.Close()
    return $true
  } catch {
    return $false
  }
}

function Get-LogStaleMinutes([string]$path) {
  if (-not (Test-Path $path)) { return [double]::PositiveInfinity }
  return ((Get-Date) - (Get-Item $path).LastWriteTime).TotalMinutes
}

$lockFree = Test-LockFree $LockPath
$staleMin = Get-LogStaleMinutes $LogPath
$unhealthy = $lockFree -or ($staleMin -gt $StaleMinutes)

if (-not $unhealthy) {
  # Cheap heartbeat so a gap in this file is itself informative (watchdog task
  # not firing at all looks different from "everything healthy").
  $counterPath = 'C:\ogrl\watchdog.counter'
  $n = 0
  if (Test-Path $counterPath) { [int]::TryParse((Get-Content $counterPath -Raw), [ref]$n) | Out-Null }
  $n++
  Set-Content -Path $counterPath -Value $n
  if ($n % $HeartbeatEveryN -eq 0) {
    Write-Log ("ok - lock held, log fresh ({0:F1} min old)" -f $staleMin)
  }
  exit 0
}

$reason = if ($lockFree) { "lock file free (supervisor process gone)" } else { ("log stale {0:N1} min (threshold {1})" -f $staleMin, $StaleMinutes) }
Write-Log "UNHEALTHY: $reason -- recovering"

$lastLine = ""
try { $lastLine = (Get-Content $LogPath -Tail 1 -ErrorAction SilentlyContinue) } catch {}
if ($lastLine) { Write-Log "last log line before recovery: $lastLine" }

# Clear stale claimants before asking Task Scheduler to relaunch, so the
# relaunch is not fighting a hung-but-still-open lock holder or orphaned
# workers waiting on a dead shm peer.
$killed = @()
Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like '*run_forever.bat*' } |
  ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    $killed += "cmd.exe(run_forever) pid=$($_.ProcessId)"
  }
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like '*train_vec.py*' } |
  ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    $killed += "python.exe(train_vec) pid=$($_.ProcessId)"
  }
$overgrowthKilled = (Get-Process -Name Overgrowth -ErrorAction SilentlyContinue | Measure-Object).Count
Start-Process -FilePath taskkill.exe -ArgumentList '/F', '/IM', 'Overgrowth.exe' -WindowStyle Hidden -Wait -ErrorAction SilentlyContinue
if ($overgrowthKilled -gt 0) { $killed += "Overgrowth.exe x$overgrowthKilled" }
Write-Log ("killed: " + ($(if ($killed.Count -gt 0) { $killed -join ', ' } else { '(nothing to kill)' })))

Start-Sleep -Seconds 3

if (-not (Test-LockFree $LockPath)) {
  Write-Log "WARNING: lock still held after cleanup -- something else holds it, not retrying this cycle"
  exit 1
}

# Let Task Scheduler own the actual (re)launch. OGRL_Supervisor's own
# restart-on-failure setting normally beats us to a plain process exit; this
# call is what recovers a hang, and is a harmless no-op if the task is
# already (re)running -- MultipleInstances=IgnoreNew on that task makes a
# double-launch impossible, which is the exact two-supervisors failure mode
# from 2026-09-08 this design must not reintroduce.
try {
  Start-ScheduledTask -TaskName $SupervisorTask
  Write-Log "requested Start-ScheduledTask $SupervisorTask"
} catch {
  Write-Log "ERROR requesting task start: $($_.Exception.Message)"
  exit 1
}
