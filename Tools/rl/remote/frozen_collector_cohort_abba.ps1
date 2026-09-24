param(
  [string] $RepoRoot = 'C:\ogrl\overgrowthRL_clean',
  [string] $Python = 'C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe',
  [string] $Checkpoint = 'C:\ogrl\overgrowthRL_clean\Tools\rl\ppo\checkpoints\run24_opt_smoke_20260921.pt',
  [string] $Engine = 'C:\ogrl\optimization\BuildWinDet232Fixed_20260923\Release\Overgrowth.exe',
  [string] $ArtifactRoot = ("C:\ogrl\cohort_screen_" + (Get-Date -Format 'yyyyMMdd_HHmmss')),
  [int] $Workers = 18,
  [int] $WarmupSeconds = 20,
  [int] $MeasureSeconds = 60,
  [int] $Seed = 2026092403,
  [double] $CohortWaitMs = 5.0,
  [int] $LaunchWaveSize = 1
)

$ErrorActionPreference = 'Stop'
$expectedCheckpoint = '1f98963795cb5d1123ee5ad8a51df870df917b387205f3d51e0c36635ce4d88d'
$expectedEngine = '9b2d423347284dfa7ebf99a8e65ec5ccf80d92896c623720d5c789d0545b5e9e'
$levels = @(
  'arenas/t_train_101.xml', 'arenas/t_train_102.xml', 'arenas/t_train_103.xml',
  'arenas/t_train_104.xml', 'arenas/t_train_105.xml', 'arenas/t_train_106.xml'
)
$arms = @(
  @{ name='A1_legacy'; minReady=1; waitMs=0.5; family='legacy' },
  @{ name='B1_cohort4'; minReady=4; waitMs=$CohortWaitMs; family='cohort4' },
  @{ name='B2_cohort4'; minReady=4; waitMs=$CohortWaitMs; family='cohort4' },
  @{ name='A2_legacy'; minReady=1; waitMs=0.5; family='legacy' }
)

function Get-Mean([double[]] $Values) {
  if (-not $Values -or $Values.Count -eq 0) { return $null }
  $total = 0.0
  foreach ($value in $Values) { $total += $value }
  return $total / $Values.Count
}

if (Test-Path $ArtifactRoot) { throw "Refusing to overwrite existing artifact root: $ArtifactRoot" }
if (-not (Test-Path $RepoRoot)) { throw "Missing checkout: $RepoRoot" }
if (-not (Test-Path $Python)) { throw "Missing Python: $Python" }
if (-not (Test-Path $Checkpoint)) { throw "Missing checkpoint: $Checkpoint" }
if (-not (Test-Path $Engine)) { throw "Missing engine: $Engine" }
if (Get-Process Overgrowth -ErrorAction SilentlyContinue) { throw 'Overgrowth.exe is already running' }

$checkpointBefore = (Get-FileHash $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
$engineBefore = (Get-FileHash $Engine -Algorithm SHA256).Hash.ToLowerInvariant()
if ($checkpointBefore -ne $expectedCheckpoint) { throw "Checkpoint hash mismatch: $checkpointBefore" }
if ($engineBefore -ne $expectedEngine) { throw "Stock engine hash mismatch: $engineBefore" }

New-Item -ItemType Directory -Path $ArtifactRoot | Out-Null
Push-Location $RepoRoot
$savedBinary = $env:OGRL_BINARY
$savedPriority = $env:OGRL_ENGINE_PRIORITY
$savedAffinity = $env:OGRL_ENGINE_AFFINITY
$savedLevelOffset = $env:OGRL_LEVEL_OFFSET
$savedLaunchWave = $env:OGRL_LAUNCH_WAVE_SIZE
$records = @()
$failure = $null
$started = (Get-Date).ToString('o')
try {
  $env:OGRL_BINARY = $Engine
  $env:OGRL_ENGINE_PRIORITY = 'above'
  $env:OGRL_ENGINE_AFFINITY = '0xFFF'
  $env:OGRL_LEVEL_OFFSET = '0'
  $env:OGRL_LAUNCH_WAVE_SIZE = [string]$LaunchWaveSize

  $sourceCommit = (git rev-parse HEAD).Trim()
  $sourceDirty = @(git status --porcelain)
  $pythonVersion = (& $Python -c "import sys,numpy,torch; print(sys.version.split()[0]+' numpy='+numpy.__version__+' torch='+torch.__version__)" | Out-String).Trim()
  $help = (& $Python 'Tools\rl\frozen_collector_sweep.py' --help 2>&1 | Out-String)
  foreach ($option in @('--opponents','--difficulty','--soft-reset','--hard-reset-every','--capture-engine-logs')) {
    if ($help -notmatch [regex]::Escape($option)) { throw "Checkout lacks required collector option $option; sync the reviewed source commit first" }
  }

  foreach ($arm in $arms) {
    if (Get-Process Overgrowth -ErrorAction SilentlyContinue) { throw "An Overgrowth engine was present before $($arm.name)" }
    if ((Get-FileHash $Engine -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedEngine) {
      throw "Engine hash changed before $($arm.name)"
    }
    $armDir = Join-Path $ArtifactRoot $arm.name
    New-Item -ItemType Directory -Path $armDir | Out-Null
    $tag = 'cohort_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + $arm.name
    $out = Join-Path $armDir 'result.json'
    $stdout = Join-Path $armDir 'stdout.log'
    $stderr = Join-Path $armDir 'stderr.log'
    $pointStart = Get-Date
    $arguments = @(
      'Tools\rl\frozen_collector_sweep.py',
      '--repo-root', $RepoRoot,
      '--checkpoint', $Checkpoint,
      '--levels'
    ) + $levels + @(
      '--workers', $Workers,
      '--k-standby', '0',
      '--collector', 'async',
      '--rollout-steps', '8',
      '--min-ready-batch', $arm.minReady,
      '--max-ready-wait-ms', $arm.waitMs,
      '--torch-threads', '2',
      '--torch-interop-threads', '1',
      '--frame-stack', '4',
      '--act-period', '4',
      '--max-episode-steps', '1200',
      '--opponents', '3',
      '--difficulty', '1.0',
      '--soft-reset',
      '--hard-reset-every', '50',
      '--capture-engine-logs',
      '--warmup-seconds', $WarmupSeconds,
      '--measure-seconds', $MeasureSeconds,
      '--seed', $Seed,
      '--shm-tag', $tag,
      '--out', $out
    )
    "ARM_STARTED $($arm.name) $($pointStart.ToString('o')) min_ready=$($arm.minReady) wait_ms=$($arm.waitMs)" |
      Add-Content (Join-Path $ArtifactRoot 'timeline.txt')
    Write-Output "ARM_STARTED $($arm.name) $(Get-Date -Format o)"
    & $Python -u @arguments 1> $stdout 2> $stderr
    $exitCode = $LASTEXITCODE
    $pointEnd = Get-Date
    if (-not (Test-Path $out)) { throw "Missing output for $($arm.name), Python exit=$exitCode" }
    $point = Get-Content -Raw $out | ConvertFrom-Json
    $evidence = $point.engine_character_log_evidence
    $observedMaps = @($evidence.observed_maps)
    $widths = @{}
    $evidence.engines | Group-Object observed_opponents_from_notice_logs |
      ForEach-Object { $widths[[string]$_.Name] = $_.Count }
    $record = [pscustomobject]@{
      arm=$arm.name
      collector='async'
      min_ready_batch=$arm.minReady
      max_ready_wait_ms=$arm.waitMs
      started_at=$pointStart.ToString('o')
      ended_at=$pointEnd.ToString('o')
      decisions_per_second=$point.decisions_per_second
      measured_seconds=$point.measured_seconds
      transitions=$point.transitions
      mean_ready_batch=$point.mean_ready_batch
      p10_ready_batch=$point.p10_ready_batch
      p90_ready_batch=$point.p90_ready_batch
      mean_ready_wait_ms=$point.mean_ready_wait_ms
      p90_ready_wait_ms=$point.p90_ready_wait_ms
      engine_count=$evidence.observed_engine_count
      actor_width_histogram=$widths
      observed_maps=$observedMaps
      characters_valid=$point.characters_valid
      checkpoint_step=$point.checkpoint_global_step
      optimizer_created=$point.optimizer_created
      checkpoint_written=$point.checkpoint_written
      python_exit_code=$exitCode
      result_json=$out
    }
    $records += $record
    $record | ConvertTo-Json -Compress | Write-Output

    if ($exitCode -ne 0 -or -not $point.characters_valid -or -not $evidence.valid) {
      throw "Actor/map evidence or collector process failed for $($arm.name); see $out and $stderr"
    }
    if ($evidence.observed_engine_count -ne $Workers -or $widths.Count -ne 1 -or $widths['3'] -ne $Workers) {
      throw "Expected $Workers engines with four actors each for $($arm.name)"
    }
    if ($observedMaps.Count -ne 6) { throw "Expected all six maps in $($arm.name), saw $($observedMaps.Count)" }
    if ($point.optimizer_created -or $point.checkpoint_written) { throw "Frozen screen modified training state in $($arm.name)" }
    if ((Get-FileHash $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedCheckpoint) {
      throw "Checkpoint changed after $($arm.name)"
    }
    if (Get-Process Overgrowth -ErrorAction SilentlyContinue) { throw "Engine survived $($arm.name)" }
    Write-Output "ARM_COMPLETED $($arm.name) $(Get-Date -Format o)"
  }
} catch {
  $failure = $_.Exception.Message
  Write-Output "COHORT_SCREEN_FAILURE $failure"
} finally {
  $checkpointAfter = (Get-FileHash $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
  $engineAfter = (Get-FileHash $Engine -Algorithm SHA256).Hash.ToLowerInvariant()
  $ended = (Get-Date).ToString('o')
  if ($null -eq $savedBinary) { Remove-Item Env:OGRL_BINARY -ErrorAction SilentlyContinue } else { $env:OGRL_BINARY = $savedBinary }
  if ($null -eq $savedPriority) { Remove-Item Env:OGRL_ENGINE_PRIORITY -ErrorAction SilentlyContinue } else { $env:OGRL_ENGINE_PRIORITY = $savedPriority }
  if ($null -eq $savedAffinity) { Remove-Item Env:OGRL_ENGINE_AFFINITY -ErrorAction SilentlyContinue } else { $env:OGRL_ENGINE_AFFINITY = $savedAffinity }
  if ($null -eq $savedLevelOffset) { Remove-Item Env:OGRL_LEVEL_OFFSET -ErrorAction SilentlyContinue } else { $env:OGRL_LEVEL_OFFSET = $savedLevelOffset }
  if ($null -eq $savedLaunchWave) { Remove-Item Env:OGRL_LAUNCH_WAVE_SIZE -ErrorAction SilentlyContinue } else { $env:OGRL_LAUNCH_WAVE_SIZE = $savedLaunchWave }
  Pop-Location

  $legacy = @($records | Where-Object { $_.arm -like '*legacy*' } | ForEach-Object { [double]$_.decisions_per_second })
  $cohort = @($records | Where-Object { $_.arm -like '*cohort4*' } | ForEach-Object { [double]$_.decisions_per_second })
  $legacyMean = if ($legacy.Count) { [math]::Round((Get-Mean ([double[]]$legacy)), 3) } else { $null }
  $cohortMean = if ($cohort.Count) { [math]::Round((Get-Mean ([double[]]$cohort)), 3) } else { $null }
  $gainPercent = if ($legacyMean -and $legacyMean -gt 0 -and $null -ne $cohortMean) {
    [math]::Round((($cohortMean / $legacyMean) - 1.0) * 100.0, 3)
  } else { $null }
  $summary = [ordered]@{
    screen='frozen-policy collector decisions/s; not PPO training throughput'
    order='A/B/B/A'
    host=$env:COMPUTERNAME
    repo=$RepoRoot
    source_commit=$(if ($sourceCommit) { $sourceCommit } else { $null })
    source_dirty_paths=$(if ($sourceDirty) { $sourceDirty } else { @() })
    python=$(if ($pythonVersion) { $pythonVersion } else { $null })
    engine_sha256=$engineAfter
    checkpoint_sha256_before=$checkpointBefore
    checkpoint_sha256_after=$checkpointAfter
    checkpoint_unchanged=($checkpointBefore -eq $checkpointAfter)
    workers=$Workers
    levels=$levels
    opponents=3
    difficulty=1.0
    soft_reset=$true
    hard_reset_every=50
    warmup_seconds=$WarmupSeconds
    measure_seconds=$MeasureSeconds
    seed=$Seed
    engine_priority='above'
    engine_affinity='0xFFF'
    launch_wave_size=$LaunchWaveSize
    legacy_mean_decisions_per_second=$legacyMean
    cohort4_mean_decisions_per_second=$cohortMean
    cohort4_change_percent=$gainPercent
    diagnostic_only=$true
    failure=$failure
    started_at=$started
    ended_at=$ended
    arms=$records
  }
  $summary | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $ArtifactRoot 'summary.json') -Encoding UTF8
  Write-Output "SUMMARY_PATH $(Join-Path $ArtifactRoot 'summary.json')"
}

if ($failure) { throw $failure }
