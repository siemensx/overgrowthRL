$py = "C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe"
$repo = "C:\ogrl\overgrowthRL"

# Wrapper batch so the task has one simple command line and its own log.
$bat = @"
@echo off
cd /d $repo
"$py" "$repo\Tools\rl\evaluate.py" --checkpoint "$repo\Tools\rl\ppo\checkpoints\run15.pt" --repo-root "$repo" --level arenas/oval_arena_1v1_unarmed.xml --frame-stack 4 --act-period 4 --episodes 200 --seed-base 5150000 --difficulty-bands 0.1,0.3,0.5,0.7,0.9,1.0 --shm-name /ogrl_evfull --device cpu --out C:\ogrl\eval_run15_windows_rebaseline.json > C:\ogrl\eval_run15_windows.log 2>&1
echo DONE %ERRORLEVEL% >> C:\ogrl\eval_run15_windows.log
"@
Set-Content -Path C:\ogrl\run_eval.bat -Value $bat -Encoding ascii

schtasks /Delete /TN OGRL_Eval /F 2>&1 | Out-Null
schtasks /Create /TN OGRL_Eval /TR "C:\ogrl\run_eval.bat" /SC ONCE /ST 23:59 /RL HIGHEST /F 2>&1 | Select-Object -Last 2
schtasks /Run /TN OGRL_Eval 2>&1 | Select-Object -Last 2
Start-Sleep -Seconds 20
"--- status ---"
schtasks /Query /TN OGRL_Eval /FO LIST 2>&1 | Select-String "Status|Last Run|Last Result"
"engines: $((Get-Process Overgrowth -ErrorAction SilentlyContinue | Measure-Object).Count)"
"log size: $(if(Test-Path C:\ogrl\eval_run15_windows.log){(Get-Item C:\ogrl\eval_run15_windows.log).Length}else{'missing'})"
