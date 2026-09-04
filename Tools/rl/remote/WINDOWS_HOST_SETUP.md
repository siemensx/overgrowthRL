# Windows training host — one-time setup

Goal: after this, the Mac (and any agent running on it) reaches the Dell with
`ssh trainer` and can measure, configure and drive it without you touching Windows again.

**You run Part 1 on the Dell. Everything after that happens from the Mac.**

---

## Part 1 — on the Dell (about 10 minutes, once)

Press `Win`, type `PowerShell`, right-click **Windows PowerShell** → **Run as administrator**.
Paste this whole block at once and press Enter.

```powershell
# --- 1. SSH server -------------------------------------------------------
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' `
  -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 `
  -ErrorAction SilentlyContinue

# --- 2. Authorise the Mac's key -----------------------------------------
# Administrator accounts use administrators_authorized_keys, NOT ~/.ssh/authorized_keys.
# The file must be ASCII (no BOM) and readable only by SYSTEM + Administrators,
# or sshd silently ignores it. Both are the usual reasons this step fails.
$key  = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIN1+8ZfBgFI3Dj21NXtVofrXFZhBbsg+uxZHr43tw/qe ogrl-mac-to-trainer'
$path = "$env:ProgramData\ssh\administrators_authorized_keys"
Add-Content -Path $path -Value $key -Encoding ascii
icacls $path /inheritance:r /grant 'Administrators:F' /grant 'SYSTEM:F'

# --- 3. Modern shell + Tailscale + tools --------------------------------
winget install --id Microsoft.PowerShell   --silent --accept-source-agreements --accept-package-agreements
winget install --id tailscale.tailscale    --silent --accept-source-agreements --accept-package-agreements
winget install --id Git.Git                --silent --accept-source-agreements --accept-package-agreements
winget install --id Python.Python.3.12     --silent --accept-source-agreements --accept-package-agreements

# Make PowerShell 7 the SSH login shell (nicer than 5.1 for everything that follows)
New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell `
  -Value 'C:\Program Files\PowerShell\7\pwsh.exe' -PropertyType String -Force

# --- 4. Never sleep ------------------------------------------------------
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT 4f971e89-eebd-4455-a8de-9e59040e7347 5ca83367-6e45-459f-a27b-476b1d01c936 0  # lid close = do nothing
powercfg /setactive SCHEME_CURRENT

# --- 5. Report what the Mac needs ---------------------------------------
Write-Host "`n=== GIVE THESE TO THE MAC ===" -ForegroundColor Green
Write-Host "user     : $env:USERNAME"
Write-Host "hostname : $env:COMPUTERNAME"
(Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
  Select-Object IPAddress, InterfaceAlias | Format-Table | Out-String).Trim()
Write-Host "sshd     : $((Get-Service sshd).Status)"
```

Then **one manual step Windows requires a human for** — bring Tailscale up and sign in:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' up
```

A browser opens; sign in with the same account you'll use on the Mac. Then:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' ip -4
```

**Send back:** the username, the Tailscale IP (100.x.x.x), and the LAN IP.

That is all you need to do on Windows.

---

## Part 2 — on the Mac (I run this)

Adds the host to `~/.ssh/config` with connection multiplexing so every subsequent agent command
is instant rather than paying a fresh TCP + auth handshake:

```
Host trainer
    HostName <tailscale-ip>
    User <windows-username>
    IdentityFile ~/.ssh/ogrl_trainer
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

Verify with `ssh trainer 'hostname; $PSVersionTable.PSVersion'`.

---

## Part 3 — sizing, which I do over SSH once it's up

Nothing here needs you. The measurements that decide the training configuration:

- **CPU identity** — `Get-CimInstance Win32_Processor`. Core Ultra 7 **165H** (6 P-cores) or
  **165U** (2 P-cores) changes the concurrency plan and the whole value case (see
  OGRL-20260904-056 §3).
- **P-core vs E-core topology** — `Get-CimInstance Win32_Processor | Select NumberOfCores,
  NumberOfLogicalProcessors`, plus `coreinfo` for the hybrid layout, so environments can be pinned
  to P-cores instead of letting Thread Director drift them onto E-cores.
- **RAM and disk** — 16 GB vs 32 GB caps concurrency; `Tools/rl/runs/` grows by gigabytes.
- **Thermal headroom** — sustained clock under load, which decides whether more environments
  actually help or just throttle the machine.
- **What is stealing the machine** — running services, startup items, Defender exclusions,
  Search indexing, OneDrive, Steam auto-start.
- **Steam + Overgrowth + mod audit** — AGENTS.md requires this before any canonical run; Workshop
  subscriptions follow the account across machines and Dynamic AI Aggression is behaviour-changing,
  not cosmetic.

---

## Troubleshooting

**`Permission denied (publickey)`** — almost always the key file. On the Dell:

```powershell
icacls "$env:ProgramData\ssh\administrators_authorized_keys"   # must list ONLY SYSTEM and Administrators
Get-Content "$env:ProgramData\ssh\administrators_authorized_keys" -Encoding Byte -TotalCount 3
# 239 187 191 means a UTF-8 BOM was written -- rewrite the file with -Encoding ascii
Get-Service sshd
```

**Connection times out** — confirm the firewall rule exists (`Get-NetFirewallRule -Name sshd`) and
that you are using the Tailscale IP if the two machines are on different networks.

**Works, then dies overnight** — the Dell slept. Re-check the `powercfg` block; on some Dell
firmware "Modern Standby" also needs disabling in BIOS.
