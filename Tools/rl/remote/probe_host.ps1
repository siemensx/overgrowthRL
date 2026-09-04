# OGRL trainer-host sizing probe. Read-only: changes nothing.
$ErrorActionPreference = 'Continue'
function Sec($t) { "`n=== $t ===" }

Sec "CPU"
$p = Get-CimInstance Win32_Processor
"name        : $($p.Name)"
"cores       : $($p.NumberOfCores) physical / $($p.NumberOfLogicalProcessors) logical"
"maxclock    : $($p.MaxClockSpeed) MHz"
"L2/L3       : $([math]::Round($p.L2CacheSize/1024,1)) MB / $([math]::Round($p.L3CacheSize/1024,1)) MB"

Sec "OS / memory"
$os = Get-CimInstance Win32_OperatingSystem
"os          : $($os.Caption) build $($os.BuildNumber)"
"ram total   : $([math]::Round($os.TotalVisibleMemorySize/1MB,1)) GB"
"ram free    : $([math]::Round($os.FreePhysicalMemory/1MB,1)) GB"
Get-CimInstance Win32_PhysicalMemory | ForEach-Object {
  "  module    : $([math]::Round($_.Capacity/1GB)) GB @ $($_.Speed) MT/s"
}

Sec "power / chassis"
"model       : $((Get-CimInstance Win32_ComputerSystem).Model)"
"bios        : $((Get-CimInstance Win32_BIOS).SMBIOSBIOSVersion)"
$b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($b) { "battery     : $($b.BatteryStatus) (2 = on AC)" }
"scheme      : $((powercfg /getactivescheme) -join '')"

Sec "disks"
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
  "  $($_.DeviceID) $([math]::Round($_.Size/1GB)) GB total, $([math]::Round($_.FreeSpace/1GB)) GB free"
}

Sec "logical processor topology (P vs E cores)"
# Efficiency class: 1 = performance core, 0 = efficiency core on Intel hybrid.
try {
  $cs = Get-CimInstance -Namespace root\cimv2 -ClassName Win32_Processor
  "note      : Win32 does not expose efficiency class; see coreinfo section below"
} catch {}

Sec "toolchain present"
foreach ($exe in 'git','python','py','cmake','pwsh','rsync','cl','clang-cl','ninja') {
  $c = Get-Command $exe -ErrorAction SilentlyContinue
  if ($c) { "  {0,-10} {1}" -f $exe, $c.Source } else { "  {0,-10} MISSING" -f $exe }
}

Sec "Visual Studio / MSVC"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) { & $vswhere -latest -products * -property displayName,installationVersion }
else { "vswhere MISSING - no Visual Studio installer present" }

Sec "Steam / Overgrowth"
$steamPaths = @(
  "${env:ProgramFiles(x86)}\Steam\steamapps\common\Overgrowth",
  "$env:ProgramFiles\Steam\steamapps\common\Overgrowth",
  "C:\Steam\steamapps\common\Overgrowth"
)
$found = $steamPaths | Where-Object { Test-Path $_ }
if ($found) { "overgrowth  : $found" } else { "overgrowth  : NOT FOUND in default locations" }
$ws = "${env:ProgramFiles(x86)}\Steam\steamapps\workshop\content\25000"
if (Test-Path $ws) {
  "workshop mods installed:"
  Get-ChildItem $ws -Directory | ForEach-Object { "  mod id $($_.Name)" }
} else { "workshop    : none at $ws" }

Sec "Defender status"
try {
  $d = Get-MpPreference
  "realtime disabled : $((Get-MpComputerStatus).RealTimeProtectionEnabled)"
  "path exclusions   : $(if ($d.ExclusionPath) { $d.ExclusionPath -join '; ' } else { '(none)' })"
  "process exclusions: $(if ($d.ExclusionProcess) { $d.ExclusionProcess -join '; ' } else { '(none)' })"
} catch { "Defender query failed: $_" }

Sec "top 15 processes by CPU time"
Get-Process | Sort-Object CPU -Descending | Select-Object -First 15 |
  Format-Table -AutoSize Name, Id, @{N='CPU(s)';E={[math]::Round($_.CPU,1)}}, @{N='RAM(MB)';E={[math]::Round($_.WorkingSet64/1MB)}} |
  Out-String

Sec "auto-start services running (non-Microsoft-critical)"
Get-CimInstance Win32_Service -Filter "StartMode='Auto' AND State='Running'" |
  Select-Object -ExpandProperty Name | Sort-Object | ForEach-Object { "  $_" }

Sec "startup items"
Get-CimInstance Win32_StartupCommand | ForEach-Object { "  $($_.Name) -> $($_.Command)" }
