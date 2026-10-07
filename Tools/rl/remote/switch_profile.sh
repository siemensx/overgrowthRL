#!/usr/bin/env bash
# One-click switch of what the Windows trainer trains (OGRL-20261007-008).
#
#   Tools/rl/remote/switch_profile.sh <profile>           # e.g. run39_personas_blind
#   Tools/rl/remote/switch_profile.sh <profile> --dry-run # show every step, change nothing
#   Tools/rl/remote/switch_profile.sh --status            # what is training now
#   Tools/rl/remote/switch_profile.sh --stop              # stop gracefully, start nothing
#   HOST=trainer-ts Tools/rl/remote/switch_profile.sh ... # over Tailscale instead of the LAN
#
# Steps, each verified before the next:
#   1. the trainer is reachable and its v6 checkout fast-forwards cleanly to origin
#   2. the running run (any OGRL_Train* task in state Running) is stopped GRACEFULLY via its
#      control.json -- final checkpoint saved -- and we wait for train_vec to exit (never kill -9)
#   3. every map the profile needs and the level script are present on the trainer: missing maps are
#      copied from this Mac's Steam tree (MD5-checked); an old level script is regenerated there
#   4. run_profile.py --check passes on the trainer (it also freezes the fork's seed checkpoint)
#   5. every other OGRL_Train* task is disabled, OGRL_Train is (re)pointed at the profile and started
#   6. after ~4 minutes, the new run's metrics.jsonl must exist and be fresh
set -uo pipefail
HOST="${HOST:-trainer-lan}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WINPS="$HERE/winps.sh"
REPO='C:\ogrl\overgrowthRL_v6'
PY='C:\Users\pavlov\AppData\Local\Programs\Python\Python312\python.exe'
PROFILE="${1:-}"; DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
[ -z "$PROFILE" ] && { sed -n '2,20p' "$0"; exit 2; }

ps1() { "$WINPS" "$HOST" 2>/dev/null | tr -d '\r' | sed 's/#< CLIXML.*//'; }
say() { printf '\n== %s\n' "$*"; }
die() { printf 'ABORT: %s\n' "$*" >&2; exit 1; }
run() { if [ $DRY = 1 ]; then echo "[dry-run] would: $1"; cat >/dev/null; else ps1; fi; }

host_ip=$(ssh -G "$HOST" 2>/dev/null | awk '/^hostname /{print $2}')
nc -z -G 5 "$host_ip" 22 >/dev/null 2>&1 || die "$HOST ($host_ip) not reachable on port 22 (ping proves nothing; try HOST=trainer-ts)"

running_runs() {
cat <<'EOF' | ps1
$a = Get-Content C:\ogrl\active_profile.json -ErrorAction SilentlyContinue | ConvertFrom-Json
Get-ScheduledTask | Where-Object { $_.TaskName -like 'OGRL_Train*' -and $_.State -eq 'Running' } | ForEach-Object {
  $cmd = ($_.Actions | ForEach-Object { $_.Arguments }) -join ' '
  if ($cmd -match 'win_train\.bat\s+(\S+)') { $rid = if ($a) { $a.run_id } else { $Matches[1] } }
  elseif ($cmd -match '(win_run\S+\.bat)') { $b = Get-Content ("C:\ogrl\overgrowthRL_v6\Tools\rl\" + $Matches[1]) -ErrorAction SilentlyContinue | Select-String '^set RUN=(\S+)' ; $rid = $b.Matches[0].Groups[1].Value }
  "RUNNING " + $_.TaskName + " " + $rid
}
"TRAINERS " + @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*train_vec.py*' }).Count
EOF
}

if [ "$PROFILE" = "--status" ]; then running_runs; exit 0; fi

if [ "$PROFILE" != "--stop" ]; then
  [ -f "$HERE/profiles/$PROFILE.json" ] || die "no profile Tools/rl/profiles/$PROFILE.json"
  say "1. code: this Mac's pushed branch vs the trainer's checkout"
  BRANCH=$(git -C "$HERE/../.." rev-parse --abbrev-ref HEAD)
  git -C "$HERE/../.." fetch -q origin "$BRANCH" && \
    [ "$(git -C "$HERE/../.." rev-parse HEAD)" = "$(git -C "$HERE/../.." rev-parse FETCH_HEAD)" ] || \
    die "$BRANCH is not pushed (local HEAD != origin) -- commit and push first; the trainer only takes code from git"
  # The trainer checkout is a DETACHED HEAD at a commit of $BRANCH; it moves only forward.
  out=$(cat <<EOF | ps1
cd $REPO
\$d = git status --porcelain --untracked-files=no
if (\$d) { "DIRTY: " + (\$d -join '; ') }
git fetch -q origin $BRANCH 2>&1 | Out-Null
git merge-base --is-ancestor HEAD FETCH_HEAD; if (\$LASTEXITCODE -ne 0) { "NOT-FORWARD: trainer HEAD is not an ancestor of origin/$BRANCH" }
"trainer at " + (git rev-parse --short HEAD) + "; will move forward over: " + ((git log --oneline HEAD..FETCH_HEAD) -join " | ")
EOF
)
  echo "$out"
  echo "$out" | grep -q '^DIRTY' && die "trainer checkout has local changes"
  echo "$out" | grep -q '^NOT-FORWARD' && die "trainer checkout is not behind $BRANCH -- inspect it by hand"
fi

say "2. stop the running run gracefully"
info=$(running_runs); echo "$info"
while read -r _ task rid; do
  [ -z "$rid" ] && die "task $task is running but its run id could not be read; stop it by hand with stop_run.sh"
  [ "$rid" = "$(python3 -c "import json;print(json.load(open('$HERE/profiles/$PROFILE.json'))['run_id'])" 2>/dev/null)" ] && \
    { echo "$rid is already the profile's run -- nothing to switch"; exit 0; }
  echo "stopping $rid ($task)"
  run "write {\"command\":\"stop\"} to $rid control.json and wait" <<EOF
Set-Content -Path "$REPO\Tools\rl\runs\\$rid\control.json" -Value '{"command": "stop"}'
\$t0 = Get-Date
while (@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { \$_.CommandLine -like '*train_vec.py*' }).Count -gt 0) {
  if (((Get-Date) - \$t0).TotalMinutes -gt 25) { "STILL RUNNING after 25 min"; exit 1 }
  Start-Sleep -Seconds 15
}
"stopped after " + [int]((Get-Date) - \$t0).TotalSeconds + " s"
Disable-ScheduledTask -TaskName "$task" | Out-Null
"disabled task $task"
EOF
done < <(echo "$info" | grep '^RUNNING')
[ "$PROFILE" = "--stop" ] && exit 0

say "2b. move the trainer checkout to origin/$BRANCH"
run "git checkout --detach origin/$BRANCH" <<EOF
cd $REPO
git fetch -q origin $BRANCH 2>&1 | Out-Null
git checkout -q --detach FETCH_HEAD 2>&1 | Out-Null
"trainer now at " + (git rev-parse --short HEAD)
EOF

say "3. maps and level script on the trainer"
need=$(cd "$HERE" && python3 -c "
import sys; sys.argv=['x','$PROFILE']; import run_profile as r
print(' '.join(r.load_profile('$PROFILE')['levels']))")
DATA_MAC=$(cd "$HERE" && python3 -c 'import paths;print(paths.data_dir())')
DATA_WIN=$(printf '%s\n' "cd $REPO; & '$PY' -c \"import sys; sys.path.insert(0, 'Tools/rl'); import paths; print(paths.data_dir())\"" | ps1 | tail -1)
case "$DATA_WIN" in [A-Z]:\\*) ;; *) die "could not resolve the trainer's Data directory (got: $DATA_WIN)";; esac
echo "trainer Data: $DATA_WIN"
missing=$(for l in $need; do printf '%s\n' "if (-not (Test-Path '$DATA_WIN\\Levels\\${l//\//\\}')) { '$l' }"; done | ps1)
for l in $missing; do
  [ -f "$DATA_MAC/Levels/$l" ] || die "map $l missing on BOTH hosts -- generate it on the Mac first"
  echo "copying $l to the trainer"
  if [ $DRY = 0 ]; then
    # via a staging folder: the Steam path has spaces and parentheses, which scp to Windows mangles
    printf '%s\n' 'New-Item -ItemType Directory -Force C:\ogrl\maps_staging | Out-Null' | ps1 >/dev/null
    scp -q "$DATA_MAC/Levels/$l" "$HOST:C:/ogrl/maps_staging/$(basename "$l")" || die "scp of $l failed"
    m1=$(md5 -q "$DATA_MAC/Levels/$l")
    m2=$(printf '%s\n' "Copy-Item -Force 'C:\ogrl\maps_staging\\$(basename "$l")' '$DATA_WIN\\Levels\\${l//\//\\}'; (Get-FileHash -Algorithm MD5 '$DATA_WIN\\Levels\\${l//\//\\}').Hash.ToLower()" | ps1 | tail -1)
    [ "$m1" = "$m2" ] || die "MD5 mismatch for $l ($m1 vs $m2)"
  fi
done
req=$(python3 -c "import json;print(json.load(open('$HERE/profiles/$PROFILE.json')).get('requires_scenario') or '')")
if [ -n "$req" ]; then
  has=$(printf '%s\n' "Select-String -Path '$DATA_WIN\\Scripts\\arena_level_1v1_unarmed.as' -Pattern '$req' -Quiet" | ps1 | tail -1)
  if [ "$has" != "True" ]; then
    echo "level script predates $req -- regenerating it on the trainer (backward compatible: persona 0 = old opponent)"
    run "gen_1v1_scenario.py on the trainer" <<EOF
cd $REPO; & '$PY' Tools\rl\gen_1v1_scenario.py 2>&1 | Select-Object -First 1
EOF
  fi
fi

say "4. preflight on the trainer"
pre=$(printf '%s\n' "cd $REPO; & '$PY' Tools\rl\run_profile.py $PROFILE --check 2>&1" | ps1)
echo "$pre"
echo "$pre" | grep -q '"problems": \[\]' || { [ $DRY = 1 ] && echo "[dry-run] preflight would fail as above" || die "preflight failed"; }

say "5. point OGRL_Train at $PROFILE and start it"
run "register + start OGRL_Train -> win_train.bat $PROFILE" <<EOF
Get-ScheduledTask | Where-Object { \$_.TaskName -like 'OGRL_Train_*' -and \$_.State -ne 'Disabled' } | ForEach-Object { Disable-ScheduledTask -TaskName \$_.TaskName | Out-Null; "disabled " + \$_.TaskName }
\$a = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c $REPO\Tools\rl\win_train.bat $PROFILE"
\$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName OGRL_Train -Action \$a -Settings \$s -RunLevel Limited -Force | Out-Null
Start-ScheduledTask -TaskName OGRL_Train
"OGRL_Train: " + (Get-ScheduledTask -TaskName OGRL_Train).State
EOF

[ $DRY = 1 ] && exit 0
say "6. verify (waiting 4 minutes for the first update)"
rid=$(python3 -c "import json;print(json.load(open('$HERE/profiles/$PROFILE.json'))['run_id'])")
sleep 240
cat <<EOF | ps1
\$m = "$REPO\Tools\rl\runs\\$rid\metrics.jsonl"
if (Test-Path \$m) { \$age = [int]((Get-Date) - (Get-Item \$m).LastWriteTime).TotalSeconds; "metrics.jsonl age \$age s" } else { "no metrics.jsonl yet -- check C:\ogrl\\$rid.log and C:\ogrl\\$rid.out" }
Get-Content C:\ogrl\\$rid.log -Tail 3
"engines: " + @(Get-Process Overgrowth -ErrorAction SilentlyContinue).Count
EOF
