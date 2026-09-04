# Remote operation of the Windows trainer host

The Mac authors and evaluates; the Windows host (`ssh trainer`) grinds long
training runs unattended. See AGENTS.md "Two-host architecture and remote
operation" for the binding rules.

| script | what it does |
|---|---|
| `WINDOWS_HOST_SETUP.md` | one-time Windows setup: OpenSSH, key auth, Tailscale, power policy |
| `probe_host.ps1` | read-only sizing: CPU/topology, RAM, disk, toolchain, mods, Defender, processes |
| `optimize_host.ps1` | power scheme, long paths, Defender exclusions, kill apps, strip startup, disable SysMain/WSearch/DoSvc/DiagTrack |
| `install_toolchain.ps1` | VS 2022 Build Tools with the VCTools workload |
| `schedule_job.ps1` | template: run a long job under a Scheduled Task so it survives SSH drops and reboots |
| `sync_artifacts.sh` | pull checkpoints and run telemetry back to the Mac (also the only backup) |

## Why Scheduled Tasks and not a detached process

`Start-Process -WindowStyle Hidden` together with `-RedirectStandardOutput`
exits immediately and silently — observed 2026-09-04, producing a dead process
and two zero-byte log files. Windows has no usable tmux either. A Scheduled
Task with `/RL HIGHEST` and a wrapper `.bat` that redirects its own output is
the reliable pattern: it survives SSH disconnects, user logoff and reboots,
which a weeks-long training run must.

## Quoting

Drive the host with **uploaded `.ps1` files**, not inline
`ssh trainer 'powershell -Command "..."'`. Quoting through zsh → ssh →
PowerShell mangles nested strings and has cost several debugging rounds.
Two further traps: PowerShell's built-in `h` alias is `Get-History`, so a
helper function named `H` fails confusingly; and `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`
must be passed inside a PowerShell argument array or it arrives as `3`.
