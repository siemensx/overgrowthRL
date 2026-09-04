# Installs the C++ toolchain needed to build the Overgrowth engine on Windows.
# VS Build Tools is a multi-GB download; expect 10-25 minutes.
$ErrorActionPreference = 'Continue'
$log = "$env:TEMP\ogrl_toolchain.log"
"start $(Get-Date -Format o)" | Out-File $log

$args = '--wait --quiet --norestart ' +
        '--add Microsoft.VisualStudio.Workload.VCTools ' +
        '--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ' +
        '--add Microsoft.VisualStudio.Component.Windows11SDK.22621 ' +
        '--includeRecommended'

"invoking winget for VS Build Tools..." | Tee-Object -Append $log
winget install --id Microsoft.VisualStudio.2022.BuildTools `
  --silent --accept-package-agreements --accept-source-agreements `
  --override $args 2>&1 | Tee-Object -Append $log

"exit code: $LASTEXITCODE" | Tee-Object -Append $log
"done $(Get-Date -Format o)" | Tee-Object -Append $log
