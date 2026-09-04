# OGRL trainer-host optimisation. Idempotent; safe to re-run.
$ErrorActionPreference = 'Continue'
function Sec($t) { "`n=== $t ===" }

Sec "power: high performance, never sleep"
# Ultimate Performance if available, else High Performance.
$ult = 'e9a42b02-d5df-448d-aa00-03f14749eb61'
powercfg /duplicatescheme $ult 2>&1 | Out-Null
$schemes = powercfg /list
if ($schemes -match $ult) { powercfg /setactive $ult; "activated Ultimate Performance" }
else { powercfg /setactive SCHEME_MIN; "activated High Performance" }
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /change disk-timeout-ac 0
# lid close on AC = do nothing
powercfg /setacvalueindex SCHEME_CURRENT 4f971e89-eebd-4455-a8de-9e59040e7347 5ca83367-6e45-459f-a27b-476b1d01c936 0
powercfg /setactive SCHEME_CURRENT
"active: " + ((powercfg /getactivescheme) -join '')

Sec "long paths (deep CMake build trees)"
New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1 -PropertyType DWord -Force | Out-Null
"LongPathsEnabled = 1"

Sec "Defender exclusions"
$paths = @('C:\ogrl', 'C:\Program Files (x86)\Steam\steamapps\common\Overgrowth')
foreach ($p in $paths) { Add-MpPreference -ExclusionPath $p -ErrorAction SilentlyContinue; "  path: $p" }
$procs = @('python.exe','Overgrowth.exe','cmake.exe','ninja.exe','cl.exe','link.exe','git.exe')
foreach ($x in $procs) { Add-MpPreference -ExclusionProcess $x -ErrorAction SilentlyContinue; "  proc: $x" }
$d = Get-MpPreference
"realtime protection enabled : $((Get-MpComputerStatus).RealTimeProtectionEnabled)"
"exclusion paths now         : $($d.ExclusionPath -join '; ')"

Sec "close unneeded applications"
$kill = @('chrome','msedge','ChatGPT','Telegram','Discord','RazerAppEngine','Razer Synapse 3',
          'steam','steamwebhelper','OneDrive','LogiDownloadAssistant','claude','Spotify','WavesSvc64')
foreach ($n in $kill) {
  $ps = Get-Process -Name $n -ErrorAction SilentlyContinue
  if ($ps) { $ps | Stop-Process -Force -ErrorAction SilentlyContinue; "  killed $n ($($ps.Count) proc)" }
}

Sec "disable startup items"
$runKeys = @('HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run',
             'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run')
$disable = 'RazerAppEngine','OneDriveSetup','Steam','GoogleChromeAutoLaunch*','Discord','Logi Download Assistant'
foreach ($k in $runKeys) {
  if (-not (Test-Path $k)) { continue }
  $props = (Get-ItemProperty $k).PSObject.Properties | Where-Object { $_.Name -notlike 'PS*' }
  foreach ($p in $props) {
    foreach ($d2 in $disable) {
      if ($p.Name -like $d2) { Remove-ItemProperty -Path $k -Name $p.Name -Force -ErrorAction SilentlyContinue; "  removed $k :: $($p.Name)" }
    }
  }
}
# Startup approved folder shortcuts
Get-ChildItem "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup" -ErrorAction SilentlyContinue |
  ForEach-Object { "  startup folder item present: $($_.Name)" }

Sec "disable background services that steal CPU/IO"
foreach ($svc in 'SysMain','WSearch','DoSvc','DiagTrack') {
  $s = Get-Service -Name $svc -ErrorAction SilentlyContinue
  if ($s) {
    Stop-Service $svc -Force -ErrorAction SilentlyContinue
    Set-Service $svc -StartupType Disabled -ErrorAction SilentlyContinue
    "  $svc -> stopped + disabled"
  }
}

Sec "result"
$os = Get-CimInstance Win32_OperatingSystem
"ram free now : $([math]::Round($os.FreePhysicalMemory/1MB,1)) GB"
Get-Process | Sort-Object CPU -Descending | Select-Object -First 8 |
  Format-Table -AutoSize Name, @{N='CPU(s)';E={[math]::Round($_.CPU,1)}}, @{N='RAM(MB)';E={[math]::Round($_.WorkingSet64/1MB)}} | Out-String
