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

#
# --- 2026-09-26 (OGRL-20260926-002) ---------------------------------------
# This watchdog was DISABLED when run27 hung, and had it been enabled as
# shipped it would have made things worse, not better: -LogPath defaulted to
# run21_win.log, a file already 5 days stale, so the very first cycle would
# have read "stale" and killed a healthy run27. Two changes:
#
#   * NO log default. -LogPath (or -HeartbeatPath) must be passed. A default
#     pointing at a dead run's log is an armed footgun, and a watchdog that
#     kills the wrong run is worse than none.
#   * Heartbeat awareness. train_vec.py now writes runs/<id>/heartbeat.json
#     at every phase boundary with a `state`, so this can tell "wedged" from
#     "deliberately paused". A run parked in the disk-low pause loop is NOT
#     unhealthy -- it is waiting on purpose and says so -- and killing it
#     would discard the tail of a multi-week run for no reason. Only a STALE
#     heartbeat (nothing written at all) means wedged.
#
# Progress signal preference, strongest first:
#   heartbeat.json  -- written even mid-update, and states WHY it is idle
#   training log    -- written once per completed update; blind to a wedge
#                      inside an update, which is precisely run27's case
# --------------------------------------------------------------------------

param(
  # Which run this watchdog owns. Recovery is SCOPED to it: without a RunId
  # the only way to clean up is a blanket kill by image name, which destroys
  # every other run on the host (see the recovery block below).
  [Parameter(Mandatory = $true)][string]$RunId,
  [string]$LockPath      = 'C:\ogrl\run_forever.lock',
  # Basename of this run's supervisor batch file, e.g. win_run27.bat. Its
  # cmd.exe is killed only when this is passed.
  [string]$SupervisorBatName = '',
  # Report the verdict and exit without killing anything. Use this to test.
  [switch]$DryRun,
  # No default: see the 2026-09-26 note above. Pass the CURRENT run's log.
  [string]$LogPath       = '',
  # runs/<run_id>/heartbeat.json. Preferred over -LogPath when present.
  [string]$HeartbeatPath = '',
  [string]$WatchdogLog   = 'C:\ogrl\watchdog.log',
  [string]$SupervisorTask = 'OGRL_Supervisor',
  [int]   $StaleMinutes  = 5,
  [int]   $HeartbeatEveryN = 12,  # ~once/hour at a 5-min trigger interval
  # States that mean "idle on purpose, do not touch". A disk-paused run is
  # waiting for a human (or for its own reaper) and must not be killed.
  [string[]]$PausedStates = @('disk_paused', 'control_paused')
)

if (-not $LogPath -and -not $HeartbeatPath) {
  Write-Error "watchdog: pass -HeartbeatPath (preferred) and/or -LogPath. There is deliberately no default: an inherited default pointing at a finished run's log makes the watchdog kill a healthy run on its first cycle (OGRL-20260926-002)."
  exit 2
}

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

function Get-HeartbeatState([string]$path) {
  # Returns @{ StaleMin; State } for runs/<id>/heartbeat.json, or $null when
  # there is no usable heartbeat (absent/unreadable/malformed) -- callers then
  # fall back to the log's mtime rather than guessing.
  if (-not $path -or -not (Test-Path $path)) { return $null }
  try {
    $j = Get-Content $path -Raw | ConvertFrom-Json
    if (-not $j.t) { return $null }
    # Trust the timestamp INSIDE the file, not the file's mtime: an atomic
    # replace can be re-touched by backup/AV without the run advancing.
    $age = ((Get-Date).ToUniversalTime() - [DateTimeOffset]::FromUnixTimeMilliseconds([long]($j.t * 1000)).UtcDateTime).TotalMinutes
    # paused_seconds/free_gb are present only on disk_paused beats. They are
    # how long the run has been WAITING -- which heartbeat age cannot tell
    # you, since a paused run keeps its heartbeat fresh by design.
    return @{
      StaleMin      = $age
      State         = [string]$j.state
      Step          = $j.global_step
      PausedMinutes = $(if ($null -ne $j.paused_seconds) { [double]$j.paused_seconds / 60.0 } else { $null })
      FreeGb        = $j.free_gb
    }
  } catch { return $null }
}

$lockFree = Test-LockFree $LockPath
$hb       = Get-HeartbeatState $HeartbeatPath
$logStale = Get-LogStaleMinutes $LogPath

if ($hb -ne $null) {
  $staleMin = $hb.StaleMin
  $progressSource = "heartbeat(state=$($hb.State), step=$($hb.Step))"
} else {
  # No heartbeat file: either an older train_vec.py, or it has not reached its
  # first phase boundary yet. Fall back to the log, which is what this script
  # always used -- weaker (blind to a wedge INSIDE an update) but not wrong.
  $staleMin = $logStale
  $progressSource = if ($HeartbeatPath) { "log (heartbeat absent/unreadable)" } else { "log" }
}

# FRESHNESS decides health; STATE only explains it. Those are deliberately
# separate axes, and conflating them is how this check goes wrong in both
# directions:
#
#   fresh + disk_paused  -> healthy. Idle on purpose AND still reporting.
#     Killing it discards the tail of a multi-week run and does not create a
#     single byte of disk space. The pause loop reaps its own orphaned
#     write-dirs and resumes by itself.
#   STALE + disk_paused  -> unhealthy. The pause loop rewrites the heartbeat
#     every 30 s, so "said disk_paused, then went silent" is a wedge INSIDE
#     the pause loop, not a pause. This is exactly run27's shape and must be
#     recovered, so state must NOT be allowed to veto a kill.
#
# Hence no state term in $unhealthy at all. An earlier version ANDed
# `-not $pausedOnPurpose` into it, which was dead weight -- $pausedOnPurpose
# required staleMin <= threshold while the clause it guarded required
# staleMin > threshold, so it could never fire -- but it read as though a
# paused state could suppress recovery, which would have been a real bug.
$pausedOnPurpose = ($hb -ne $null) -and ($PausedStates -contains $hb.State)
$fresh = ($staleMin -le $StaleMinutes)
$unhealthy = $lockFree -or -not $fresh

# A fresh pause is healthy but not NORMAL. The watchdog cannot free disk, so a
# run parked on disk_paused needs a human; run27 waited 35 h. Duration comes
# from the heartbeat's own paused_seconds -- heartbeat AGE cannot measure it,
# because a paused run keeps its heartbeat fresh on purpose. Surfaces on the
# log line only, never on the kill path.
$pausedMinutes = $(if ($pausedOnPurpose -and $null -ne $hb.PausedMinutes) { [double]$hb.PausedMinutes } else { 0.0 })

if (-not $unhealthy) {
  # Cheap heartbeat so a gap in this file is itself informative (watchdog task
  # not firing at all looks different from "everything healthy").
  $counterPath = 'C:\ogrl\watchdog.counter'
  $n = 0
  if (Test-Path $counterPath) { [int]::TryParse((Get-Content $counterPath -Raw), [ref]$n) | Out-Null }
  $n++
  Set-Content -Path $counterPath -Value $n
  if ($pausedOnPurpose) {
    # Always log this one: a paused run is a state a human should see, and it
    # is rare enough that it cannot spam the log. Escalate the wording once the
    # wait is long, because nothing automatic will end it -- the watchdog
    # cannot free disk, and killing the run would not either.
    $detail = "$progressSource, heartbeat {0:F1} min old" -f $staleMin
    if ($null -ne $hb.FreeGb) { $detail += ", {0:F2} GB free" -f [double]$hb.FreeGb }
    if ($pausedMinutes -ge 30.0) {
      Write-Log ("ATTENTION - paused on purpose for {0:F0} min and NOT self-clearing; a human must free disk ({1})" -f $pausedMinutes, $detail)
    } elseif ($pausedMinutes -gt 0.0) {
      Write-Log ("ok - paused on purpose for {0:F1} min, still reporting ({1})" -f $pausedMinutes, $detail)
    } else {
      Write-Log ("ok - paused on purpose ($detail)")
    }
  } elseif ($n % $HeartbeatEveryN -eq 0) {
    Write-Log ("ok - lock held, progress fresh ({0:F1} min old via $progressSource)" -f $staleMin)
  }
  exit 0
}

$reason = if ($lockFree) { "lock file free (supervisor process gone)" } else { ("no progress for {0:N1} min (threshold {1}) via {2}" -f $staleMin, $StaleMinutes, $progressSource) }
Write-Log "UNHEALTHY: $reason -- recovering"

$lastLine = ""
try { $lastLine = (Get-Content $LogPath -Tail 1 -ErrorAction SilentlyContinue) } catch {}
if ($lastLine) { Write-Log "last log line before recovery: $lastLine" }

# Clear stale claimants before asking Task Scheduler to relaunch, so the
# relaunch is not fighting a hung-but-still-open lock holder or orphaned
# workers waiting on a dead shm peer.
#
# SCOPED BY --run-id, not by image name (2026-09-26). This block used to do
# `taskkill /F /IM Overgrowth.exe` plus a kill of every python.exe running
# train_vec.py, which is indiscriminate: pointing the watchdog at one run and
# firing it destroys ANY other training on the host. Demonstrated the hard
# way while testing this very script -- a deliberately sandboxed test run
# (aimed at a disabled task) still killed the live run27, 24 engines and all,
# because the kill happens before the task is ever consulted. run27 lost
# ~61k steps and the supervisor recovered it in 34 s, but on a host with two
# runs the other one would simply be gone.
#
# Engines are matched by --shm-prefix, which the supervisor derives per launch
# and every engine of that trainer carries on its own command line.
if ($DryRun) {
  Write-Log "DRY RUN: would kill this run's supervisor/trainer/engines and request a restart; killing nothing"
  exit 0
}

$killed = @()
$trainerPids = @()
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like '*train_vec.py*' -and $_.CommandLine -like "*--run-id $RunId*" } |
  ForEach-Object {
    $trainerPids += $_.ProcessId
    if ($_.CommandLine -match '--shm-prefix\s+(\S+)') { $script:shmPrefix = $Matches[1] }
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    $killed += "python.exe(train_vec --run-id $RunId) pid=$($_.ProcessId)"
  }
if ($trainerPids.Count -eq 0) {
  Write-Log "note: no train_vec.py process for --run-id $RunId (already exited); continuing to clean its engines"
}
# Its supervisor cmd.exe, matched by the batch file this run actually uses.
if ($SupervisorBatName) {
  Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*$SupervisorBatName*" } |
    ForEach-Object {
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
      $killed += "cmd.exe($SupervisorBatName) pid=$($_.ProcessId)"
    }
}
# Engines belonging to THIS trainer only. A SIGKILLed engine orphans its named
# semaphore, so the relaunch must use a fresh --shm-prefix -- the supervisor
# already derives one per launch (%RANDOM%%TRIES%), which is what makes this
# safe (AGENTS.md invariant 3).
if ($shmPrefix) {
  $n = 0
  Get-CimInstance Win32_Process -Filter "Name='Overgrowth.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*$shmPrefix*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $n++ }
  if ($n) { $killed += "Overgrowth.exe x$n (shm $shmPrefix)" }
} else {
  # No prefix resolvable (trainer already gone). Reap only engines whose
  # write-dir names this run; never a blanket /IM kill.
  $n = 0
  Get-CimInstance Win32_Process -Filter "Name='Overgrowth.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*$RunId*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $n++ }
  if ($n) { $killed += "Overgrowth.exe x$n (matched $RunId)" }
  $others = (Get-CimInstance Win32_Process -Filter "Name='Overgrowth.exe'" -ErrorAction SilentlyContinue | Measure-Object).Count
  if ($others -gt 0) { Write-Log "note: leaving $others Overgrowth.exe not attributable to $RunId -- they may belong to another run" }
}
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
