# OGRL-20260924-010: hand run26 over to run27 once run26 has launched its 270M bench.
$ErrorActionPreference = 'Continue'
$repo = 'C:\ogrl\overgrowthRL_clean'
$log = 'C:\ogrl\switch27.log'
function Say($m) { "[{0}] {1}" -f (Get-Date -Format s), $m | Add-Content $log }
$metrics = "$repo\Tools\rl\runs\run26_win\metrics.jsonl"
if (-not $env:OGRL_SWITCH_AT) { $env:OGRL_SWITCH_AT = "0" }
Say "waiting for run26 >= $env:OGRL_SWITCH_AT"
while ($true) {
    $last = Get-Content $metrics -Tail 1 -ErrorAction SilentlyContinue
    if ($last) { try { $step = ($last | ConvertFrom-Json).global_step } catch { $step = 0 } }
    if ($step -ge [int64]$env:OGRL_SWITCH_AT) { break }
    Start-Sleep 60
}
Say "run26 at $step; requesting graceful stop"
'{"command":"stop"}' | Set-Content "$repo\Tools\rl\runs\run26_win\control.json" -Encoding ascii
while ((Get-ScheduledTask -TaskName 'OGRL_Train_run26').State -eq 'Running') { Start-Sleep 30 }
Say "run26 task stopped"
Disable-ScheduledTask -TaskName 'OGRL_Train_run26' | Out-Null
Get-Process Overgrowth -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 10
Copy-Item "$repo\Tools\rl\ppo\checkpoints\run26_win.pt" "$repo\Tools\rl\ppo\checkpoints\run26_win_final.pt" -Force
Say ("snapshot run26_win_final.pt sha256 " + (Get-FileHash "$repo\Tools\rl\ppo\checkpoints\run26_win_final.pt").Hash)
$a = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$repo\Tools\rl\win_run27.bat`"" -WorkingDirectory $repo
$t = New-ScheduledTaskTrigger -AtLogOn -User 'pavlov'
$p = New-ScheduledTaskPrincipal -UserId 'pavlov' -LogonType Interactive -RunLevel Limited
$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -Priority 4
Register-ScheduledTask -TaskName 'OGRL_Train_run27' -Action $a -Trigger $t -Principal $p -Settings $s -Force | Out-Null
Start-ScheduledTask -TaskName 'OGRL_Train_run27'
Say "run27 started"
