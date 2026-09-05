#!/usr/bin/env bash
# Run PowerShell on the Windows trainer without quoting pain.
# Usage:  winps.sh [host] < script.ps1
# Encodes UTF-16LE base64 -> -EncodedCommand, so quotes and newlines survive intact.
# Strips the PowerShell CLIXML progress-stream noise that SSH mixes into stderr.
set -euo pipefail
HOST="${1:-trainer-lan}"
SCRIPT=$(printf '$ProgressPreference="SilentlyContinue"\n'; cat)
ENC=$(printf '%s' "$SCRIPT" | python3 -c "import sys,base64; sys.stdout.write(base64.b64encode(sys.stdin.read().encode('utf-16-le')).decode())")
ssh -o ConnectTimeout=15 -o ServerAliveInterval=20 "$HOST" \
    "powershell -NoProfile -NonInteractive -EncodedCommand $ENC" 2>&1 \
  | { grep -vE '^#< CLIXML|^<Objs Version' || true; }
