# Verdict tests for watchdog.ps1's heartbeat-aware health check
# (OGRL-20260926-002). Every case runs with -DryRun, so nothing is killed.
#
# WHY THIS FILE EXISTS, beyond the verdict table: while testing the watchdog
# interactively on 2026-09-26 a "sandboxed" invocation -- pointed at a
# DISABLED scheduled task, which felt safe -- killed the live run27 outright,
# all 24 engines. The recovery block ran `taskkill /F /IM Overgrowth.exe` and
# killed every train_vec.py on the host BEFORE it ever consulted the task, so
# aiming it at a harmless task sandboxed nothing. run27 lost ~61k steps and
# its supervisor recovered it in 34 s. Recovery is now scoped by --run-id and
# --shm-prefix, and -DryRun exists precisely so this can never be tested
# destructively again. Never test this script without -DryRun on a host that
# is training.
#
# Heartbeat semantics the table pins:
#   state tells you WHY the loop is idle; freshness tells you WHETHER it is
#   alive. Those are independent, and both matter:
#     * fresh disk_paused   -> waiting on purpose, and still reporting. Healthy.
#       Killing it discards the tail of a multi-week run and does not create
#       disk space.
#     * STALE disk_paused   -> it announced the pause and then stopped writing
#       at all. The pause loop heartbeats every 30 s, so this is a wedge
#       INSIDE the pause loop, not a pause. Recover.
#   That distinction is the whole point: run27's 35 h hang looked identical to
#   a deadlock from outside precisely because nothing kept reporting.
#
# Run:  powershell -NoProfile -ExecutionPolicy Bypass -File test_watchdog_heartbeat.ps1 `
#           -Watchdog <path to watchdog.ps1> -SupervisorTask <any existing task>

param(
  [string]$Watchdog       = (Join-Path $PSScriptRoot 'watchdog.ps1'),
  [string]$SupervisorTask = 'OGRL_Train_run27',
  [string]$WorkDir        = (Join-Path $env:TEMP ('wdtest_' + [guid]::NewGuid().ToString('N').Substring(0,8)))
)

$ErrorActionPreference = 'Continue'
if (-not (Test-Path $Watchdog)) { throw "missing $Watchdog" }
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null

$lock = Join-Path $WorkDir 'lock'; Set-Content $lock 'x'
$log  = Join-Path $WorkDir 'train.out'; Set-Content $log 'update=1'
$wlog = Join-Path $WorkDir 'wd.log'

function New-Heartbeat([string]$state, [double]$ageMinutes) {
  $p = Join-Path $WorkDir 'hb.json'
  $t = [DateTimeOffset]::UtcNow.AddMinutes(-$ageMinutes).ToUnixTimeMilliseconds() / 1000.0
  Set-Content $p (@{ t = $t; state = $state; global_step = 301000000; pid = 1 } | ConvertTo-Json -Compress)
  return $p
}

$cases = @(
  @{ Name = 'fresh collecting';            State = 'collecting';     Age = 0.2;  Want = 'healthy'   },
  @{ Name = 'fresh updating';              State = 'updating';       Age = 0.5;  Want = 'healthy'   },
  @{ Name = 'fresh disk_paused';           State = 'disk_paused';    Age = 0.4;  Want = 'healthy'   },
  @{ Name = 'fresh control_paused';        State = 'control_paused'; Age = 0.4;  Want = 'healthy'   },
  # The run27 shape: announced, then silent. A pause heartbeats every 30s, so
  # a stale disk_paused is a wedge inside the pause loop.
  @{ Name = 'STALE disk_paused 40h';       State = 'disk_paused';    Age = 2400; Want = 'UNHEALTHY' },
  @{ Name = 'wedged collecting 40min';     State = 'collecting';     Age = 40;   Want = 'UNHEALTHY' },
  @{ Name = 'wedged updating 90min';       State = 'updating';       Age = 90;   Want = 'UNHEALTHY' }
)

$enginesBefore = (Get-CimInstance Win32_Process -Filter "Name='Overgrowth.exe'" -EA SilentlyContinue | Measure-Object).Count
$pass = 0; $fail = 0

# Hold the lock so lockFree does not short-circuit the heartbeat logic.
$fs = [System.IO.File]::Open($lock, 'Open', 'ReadWrite', 'None')
try {
  foreach ($c in $cases) {
    $hb = New-Heartbeat $c.State $c.Age
    Remove-Item $wlog -Force -EA SilentlyContinue
    & powershell -NoProfile -ExecutionPolicy Bypass -File $Watchdog `
        -RunId 'run_wdtest_nonexistent' -LockPath $lock -LogPath $log -HeartbeatPath $hb `
        -WatchdogLog $wlog -SupervisorTask $SupervisorTask -StaleMinutes 15 -DryRun *> $null
    $raw = if (Test-Path $wlog) { Get-Content $wlog -Raw } else { '' }
    $verdict = if ($raw -match 'UNHEALTHY') { 'UNHEALTHY' } else { 'healthy' }
    $ok = ($verdict -eq $c.Want)
    if ($ok) { $pass++ } else { $fail++ }
    Write-Output ("{0,-28} -> {1,-10} want {2,-10} {3}" -f $c.Name, $verdict, $c.Want, $(if ($ok) { 'PASS' } else { 'FAIL' }))
  }

  # -RunId is mandatory: without it recovery can only fall back to a blanket
  # kill by image name, which is the bug that killed run27.
  & powershell -NoProfile -ExecutionPolicy Bypass -File $Watchdog -LogPath $log -HeartbeatPath (New-Heartbeat 'collecting' 99) *> $null
  $refusedNoRunId = ($LASTEXITCODE -ne 0)
  if ($refusedNoRunId) { $pass++ } else { $fail++ }
  Write-Output ("{0,-28} -> {1}" -f 'missing -RunId refuses', $(if ($refusedNoRunId) { 'PASS' } else { 'FAIL' }))

  # And no log/heartbeat target at all must refuse, rather than inherit a
  # default pointing at some finished run's log.
  & powershell -NoProfile -ExecutionPolicy Bypass -File $Watchdog -RunId 'x' *> $null
  $refusedNoTarget = ($LASTEXITCODE -ne 0)
  if ($refusedNoTarget) { $pass++ } else { $fail++ }
  Write-Output ("{0,-28} -> {1}" -f 'no target refuses', $(if ($refusedNoTarget) { 'PASS' } else { 'FAIL' }))
} finally {
  $fs.Close()
}

$enginesAfter = (Get-CimInstance Win32_Process -Filter "Name='Overgrowth.exe'" -EA SilentlyContinue | Measure-Object).Count
Write-Output ''
Write-Output ("engines before={0} after={1} {2}" -f $enginesBefore, $enginesAfter,
              $(if ($enginesBefore -eq $enginesAfter) { '(untouched - DryRun held)' } else { 'ERROR: DryRun KILLED ENGINES' }))
if ($enginesBefore -ne $enginesAfter) { $fail++ }
Remove-Item $WorkDir -Recurse -Force -EA SilentlyContinue
Write-Output ("{0} passed, {1} failed" -f $pass, $fail)
exit $(if ($fail -gt 0) { 1 } else { 0 })
