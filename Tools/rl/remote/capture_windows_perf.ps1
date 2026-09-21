<#
Capture low-overhead Windows CPU and scheduler evidence beside a throughput
block. This is intentionally a separate process: it observes the machine and
does not change the power plan, process priority, or affinity.

Example (run only while a benchmark block is active):
  powershell -File .\capture_windows_perf.ps1 -Output C:\ogrl\abba_a.csv -Seconds 900

The performance-limit counter is the useful thermal/power signal on this
165U. A value below 100 means Windows or firmware is limiting requested
frequency; it is not evidence that a higher Windows power plan exists.
#>
param(
  [string]$Output = "C:\ogrl\windows_perf.csv",
  [int]$Seconds = 900,
  [int]$IntervalSeconds = 5
)

$parent = Split-Path -Parent $Output
if ($parent -and -not (Test-Path $parent)) {
  New-Item -ItemType Directory -Path $parent -Force | Out-Null
}

"utc,elapsed_s,cpu_time_pct,processor_performance_pct,maximum_frequency_pct,performance_limit_pct,overgrowth_count,overgrowth_working_set_mb,active_power_plan" |
  Set-Content -Path $Output -Encoding ascii

$plan = (powercfg /getactivescheme 2>$null) -join " "
$watch = [Diagnostics.Stopwatch]::StartNew()
while ($watch.Elapsed.TotalSeconds -lt $Seconds) {
  $cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
  $perf = (Get-Counter '\Processor Information(_Total)\% Processor Performance' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
  $max = (Get-Counter '\Processor Information(_Total)\% of Maximum Frequency' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
  $limit = (Get-Counter '\Processor Information(_Total)\% Performance Limit' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
  $engines = @(Get-Process Overgrowth -ErrorAction SilentlyContinue)
  $rss = ($engines | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB
  $stamp = (Get-Date).ToUniversalTime().ToString('o')
  $values = @(
    $stamp,
    ([math]::Round($watch.Elapsed.TotalSeconds, 3)),
    ([math]::Round([double]$cpu, 3)),
    ([math]::Round([double]$perf, 3)),
    ([math]::Round([double]$max, 3)),
    ([math]::Round([double]$limit, 3)),
    $engines.Count,
    ([math]::Round([double]$rss, 3)),
    ('"' + ($plan -replace '"', '""') + '"')
  )
  Add-Content -Path $Output -Value ($values -join ',') -Encoding ascii
  Start-Sleep -Seconds $IntervalSeconds
}
