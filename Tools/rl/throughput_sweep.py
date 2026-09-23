#!/usr/bin/env python3
"""End-to-end throughput sweep of the REAL trainer, one config at a time.

concurrency_sweep.py measures the collector ceiling with random actions on
the default scenario. That is not the number that matters for run21: the
live fight is 1v3 at difficulty 1.0, episodes run 600-1200 decisions, and
resets, the PPO update and the standby pool all interact. This launches
train_vec.py itself for a fixed wall time per (n_envs, k_standby) point,
resumed read-only from a real checkpoint, then reads the run's own
metrics.jsonl and reports median steps_per_second_cycle over the measurement
window (first `warmup` seconds discarded: engine startup). Policy updates are
in memory only; this probe never writes a checkpoint.

    python Tools/rl/throughput_sweep.py --resume-from <ckpt> --grid 10x2 10x4 12x2 12x4 8x2 14x2

Requests a clean trainer stop between points and, only as a bounded fallback,
terminates that launched process tree. It never kills unrelated Overgrowth.exe
processes. Writes one JSON with every point's numbers and prints a table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics as st
import subprocess
import sys
import time
from pathlib import Path


def find_existing_engines() -> list[str]:
    """Read-only process check used to refuse a sweep on a busy host."""
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist.exe", "/FI", "IMAGENAME eq Overgrowth.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        rows = csv.reader(result.stdout.splitlines())
        return [f"PID {row[1]}" for row in rows if len(row) > 1 and row[0].lower() == "overgrowth.exe"]

    result = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="], capture_output=True, text=True, check=True, timeout=10
    )
    found = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 2 or Path(fields[1]).name.lower() not in {"overgrowth", "overgrowth.exe"}:
            continue
        found.append(f"PID {fields[0]}")
    return found


def refuse_busy_engine_host() -> None:
    existing = find_existing_engines()
    if existing:
        raise RuntimeError(
            "refusing throughput sweep while Overgrowth processes already exist: "
            + ", ".join(existing)
            + ". Stop only the intended run explicitly, then retry."
        )


def wait_for_ready(log: Path, proc: subprocess.Popen, timeout: float) -> tuple[float | None, float]:
    """Wait for the post-reset all-active-worker readiness barrier."""
    started = time.monotonic()
    offset = 0
    carry = ""
    while time.monotonic() - started < timeout:
        if log.exists():
            try:
                with log.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offset)
                    chunk = fh.read()
                    offset = fh.tell()
                carry = (carry + chunk)[-2048:]
                match = re.search(r"\[RL_READY\] t=([0-9.]+)", carry)
                if match:
                    return float(match.group(1)), time.monotonic() - started
            except OSError:
                pass
        if proc.poll() is not None:
            return None, time.monotonic() - started
        time.sleep(0.25)
    return None, time.monotonic() - started


def request_stop(run_dir: Path) -> None:
    """Request the trainer's normal update-boundary exit without a checkpoint.

    `train_vec.py` only writes a final checkpoint when --checkpoint-path was
    supplied. The no-checkpoint throughput path deliberately omits that flag,
    so using its existing control file is both graceful and weight-safe.
    Writing atomically and as BOM-free UTF-8 matches the dashboard protocol.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    control = run_dir / "control.json"
    temporary = control.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"command": "stop", "t": time.time()}), encoding="utf-8")
    os.replace(temporary, control)


def stop_trainer(proc: subprocess.Popen, run_dir: Path, grace_seconds: float) -> tuple[bool, bool]:
    """Return (clean_exit, escalated); escalation is limited to our process tree."""
    if proc.poll() is not None:
        return proc.returncode == 0, False
    try:
        request_stop(run_dir)
    except OSError as stop_error:
        # A failed control write must not strand this probe's workers. Still
        # allow the bounded grace period for a naturally completing process,
        # then use the same owned-tree fallback below.
        print(f"could not write stop control for PID {proc.pid}: {stop_error}", file=sys.stderr)
    try:
        proc.wait(timeout=grace_seconds)
        return proc.returncode == 0, False
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            # /T scopes this fallback to the trainer launched below and its
            # engine children. Never use /IM Overgrowth.exe: another training
            # job or a human match may own those processes.
            try:
                subprocess.run(
                    ["taskkill.exe", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True, text=True, check=False, timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired) as kill_error:
                print(f"owned process-tree termination failed for PID {proc.pid}: {kill_error}", file=sys.stderr)
        else:
            # Popen uses start_new_session=True below, so this group contains
            # only the trainer and the engines it launched.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=20)
            return False, True
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                # taskkill /T already attempted scoped termination. Do not
                # fall back to killing a machine-wide set of engine processes.
                raise RuntimeError(
                    f"trainer process tree {proc.pid} survived taskkill /T; "
                    "refusing a machine-wide engine kill"
                )
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
            return False, True


def write_results(path: Path, results: list[dict]) -> None:
    """Atomically refresh this invocation's reserved summary file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite unfinished summary artifact: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=1)
        handle.write("\n")
    os.replace(temporary, path)


def run_point(args, n_envs: int, k_standby: int, tag: str) -> dict:
    repo = Path(args.repo_root)
    run_id = f"sweep_{tag}_n{n_envs}k{k_standby}"
    # A hard taskkill can leave a named semaphore behind on Windows.  The old
    # n/k-only prefix reused that semaphore on the next point, which made a
    # perfectly good control fail during startup and could make the following
    # point attach to stale IPC.  The run-specific prefix is deliberately
    # unique for every point, including repeated n/k comparisons.
    shm_prefix = f"/ogrl_{run_id}"
    run_dir = repo / "Tools" / "rl" / "runs" / run_id
    log = run_dir.parent / f"{run_id}.log"
    ckpt = repo / "Tools" / "rl" / "ppo" / "checkpoints" / f"{run_id}.pt"
    for artifact in (run_dir, log, ckpt):
        if artifact.exists():
            raise FileExistsError(f"refusing to overwrite prior benchmark artifact: {artifact}")
    cmd = [sys.executable, "-u", str(repo / "Tools" / "rl" / "ppo" / "train_vec.py"),
           "--repo-root", str(repo), "--levels", args.levels,
           "--shm-prefix", shm_prefix, "--n-envs", str(n_envs), "--k-standby", str(k_standby),
           "--seed", "7", "--resume-from", args.resume_from,
           "--run-id", run_id, "--total-timesteps", "4000000000",
           "--n-steps", str(args.n_steps), "--n-epochs", str(args.n_epochs), "--minibatch-size", str(args.minibatch_size),
           "--entropy-coef", "0.003", "--entropy-coef-final", "0.003", "--entropy-anneal-steps", "1000000",
           "--learning-rate", "0.0003", "--target-kl", "0.02", "--max-episode-steps", "1200",
           "--frame-stack", "4", "--act-period", "4", "--soft-reset",
           "--hard-reset-every", str(args.hard_reset_every),
           "--d-max-start", "1.0", "--d-max-cap", "1.0", "--d-step", "0.1", "--d-min", "1.0",
           "--opponents-cap", "3", "--opp-keep-solo", "0.0", "--armed-stage", "0",
           "--gate-eval-episodes", "30", "--gate-min-step-gap", "999999999999",
           "--allow-n-envs-change",
           "--collection-torch-threads", str(args.collection_threads),
           "--update-torch-threads", str(args.update_threads),
           "--torch-interop-threads", str(args.interop_threads),
           "--no-tapes", "--no-native-capture"] + args.extra
    env = dict(os.environ)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {run_id}: {args.warmup + args.measure:.0f}s ===", flush=True)
    launch_t = time.time()
    ready_at = None
    ready_wait = 0.0
    measurement_cutoff_epoch = None
    clean_stop = False
    cleanup_escalated = False
    # Reserve the telemetry directory exclusively so stale rows can never be
    # appended to or mistaken for this point's result.
    run_dir.mkdir(parents=True, exist_ok=False)
    popen_kwargs = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True
    with log.open("x", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(repo), stdout=lf, stderr=subprocess.STDOUT, env=env,
            **popen_kwargs,
        )
        stop_attempted = False
        try:
            ready_at, ready_wait = wait_for_ready(log, proc, args.ready_timeout)
            if ready_at is None:
                exited_early = True
                measurement_cutoff_epoch = time.time()
                stop_attempted = True
                clean_stop, cleanup_escalated = stop_trainer(proc, run_dir, args.stop_grace)
            else:
                try:
                    proc.wait(timeout=args.warmup + args.measure)
                    exited_early = True
                    measurement_cutoff_epoch = time.time()
                except subprocess.TimeoutExpired:
                    exited_early = False
                    measurement_cutoff_epoch = time.time()
                    stop_attempted = True
                    clean_stop, cleanup_escalated = stop_trainer(proc, run_dir, args.stop_grace)
                else:
                    clean_stop = proc.returncode == 0
        except BaseException:
            if not stop_attempted and proc.poll() is None:
                try:
                    stop_trainer(proc, run_dir, args.stop_grace)
                except Exception as cleanup_error:  # retain the original failure
                    print(f"benchmark cleanup failed for PID {proc.pid}: {cleanup_error}", file=sys.stderr)
            raise
    metrics = run_dir / "metrics.jsonl"
    rows = []
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    measurement_end = (
        min(measurement_cutoff_epoch, ready_at + args.warmup + args.measure)
        if ready_at is not None and measurement_cutoff_epoch is not None
        else None
    )
    win = [r for r in rows if ready_at is not None
           and r["t"] - ready_at >= args.warmup
           and (measurement_end is None or r["t"] <= measurement_end)]
    measured_rows = win[1:] if len(win) > 1 else []
    def col(k):
        return [r["perf"][k] for r in win if r.get("perf", {}).get(k) is not None]
    sps = col("steps_per_second_cycle")
    out = {"run_id": run_id, "n_envs": n_envs, "k_standby": k_standby, "rows_total": len(rows), "rows_measured": len(win),
           "exited_early": exited_early, "ready": ready_at is not None, "ready_at": ready_at,
           "ready_wait_seconds": ready_wait, "launch_wall_seconds": time.time() - launch_t,
           "trainer_returncode": proc.returncode, "clean_stop": clean_stop,
           "cleanup_escalated": cleanup_escalated,
           "measurement_cutoff_epoch": measurement_end}
    if sps:
        out.update({
            "sps_median": st.median(sps), "sps_mean": st.mean(sps), "sps_p10": sorted(sps)[len(sps) // 10],
            "collection_sps_median": st.median(col("steps_per_second_collection") or [0]),
            "barrier_idle_per_worker_s": st.mean(col("barrier_idle_seconds") or [0]) / n_envs,
            "cycle_s": st.median(col("cycle_seconds") or [0]),
            "pool_miss_rate": (sum(col("pool_misses")) / max(1, sum(col("pool_misses")) + sum(col("pool_hits")))) if col("pool_hits") else None,
            "recoveries": sum(r["perf"].get("recoveries", 0) for r in measured_rows),
            "valid_steps_measured": sum(r["perf"].get("valid_transition_count", 0) for r in measured_rows),
            "recovered_steps_measured": sum(r["perf"].get("recovered_transition_count", 0) for r in measured_rows),
            "steps_measured": (win[-1]["global_step"] - win[0]["global_step"]) if len(win) > 1 else 0,
            "wall_sps": ((win[-1]["global_step"] - win[0]["global_step"]) / (win[-1]["t"] - win[0]["t"])) if len(win) > 1 else 0,
            "useful_wall_sps": (sum(r["perf"].get("valid_transition_count", 0) for r in measured_rows) /
                                (win[-1]["t"] - win[0]["t"])) if len(win) > 1 else 0,
        })
    print(json.dumps(out), flush=True)
    # Checkpoints are never written by this benchmark. If one unexpectedly
    # appears, preserve it as evidence and invalidate the point; never delete it.
    out["checkpoint_written"] = ckpt.exists()
    manifest_path = run_dir / "run.json"
    try:
        out["run_manifest_status"] = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError, AttributeError):
        out["run_manifest_status"] = None
    out["valid_point"] = bool(
        out["ready"]
        and not out["exited_early"]
        and out["rows_measured"] >= 2
        and out["trainer_returncode"] == 0
        and out["clean_stop"]
        and out["run_manifest_status"] == "completed"
        and not out["checkpoint_written"]
    )
    if not out["valid_point"]:
        print(f"INVALID_POINT {run_id}: {json.dumps(out)}", file=sys.stderr, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--resume-from", required=True)
    ap.add_argument("--levels", default="arenas/t_train_101.xml,arenas/t_train_102.xml,arenas/t_train_104.xml")
    ap.add_argument("--grid", nargs="+", default=["10x2", "10x4", "12x2", "12x4", "8x2", "14x2"],
                    help="points as NxK, optionally with env overrides after a colon: "
                         "14x4:OGRL_ENGINE_PRIORITY=above,OGRL_ENGINE_AFFINITY=0xFFF")
    ap.add_argument("--warmup", type=float, default=150.0)
    ap.add_argument("--measure", type=float, default=360.0)
    ap.add_argument("--ready-timeout", type=float, default=180.0,
                    help="maximum seconds to wait for the post-reset RL_READY barrier")
    ap.add_argument("--stop-grace", type=float, default=120.0,
                    help="maximum seconds to wait for the trainer's clean update-boundary stop")
    ap.add_argument("--collection-threads", type=int, default=2,
                    help="PyTorch intra-op threads during rollout inference")
    ap.add_argument("--update-threads", type=int, default=4)
    ap.add_argument("--interop-threads", type=int, default=1,
                    help="PyTorch inter-op threads")
    ap.add_argument("--hard-reset-every", type=int, default=50,
                    help="periodic hard-reset interval for soft-reset throughput tests")
    ap.add_argument("--n-steps", type=int, default=256,
                    help="rollout horizon per environment for the PPO update")
    ap.add_argument("--n-epochs", type=int, default=1,
                    help="PPO optimization epochs per rollout")
    ap.add_argument("--minibatch-size", type=int, default=128,
                    help="PPO minibatch size")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="legacy compatibility flag; this benchmark always runs without checkpoint output")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    refuse_busy_engine_host()
    out = Path(args.out) if args.out else Path(args.repo_root) / "Tools" / "rl" / "runs" / f"throughput_sweep_{args.tag}.json"
    temporary_out = out.with_suffix(out.suffix + ".tmp")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite prior sweep summary: {out}")
    if temporary_out.exists():
        raise FileExistsError(f"unfinished sweep summary exists: {temporary_out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for g in args.grid:
        refuse_busy_engine_host()
        spec, _, envs = g.partition(":")
        n, k = (int(x) for x in spec.lower().split("x"))
        overrides = dict(kv.split("=", 1) for kv in envs.split(",") if kv)
        saved = {kk: os.environ.get(kk) for kk in overrides}
        os.environ.update(overrides)
        try:
            r = run_point(args, n, k, args.tag + ("_" + "_".join(v for v in overrides.values()) if overrides else ""))
        finally:
            for kk, vv in saved.items():
                if vv is None: os.environ.pop(kk, None)
                else: os.environ[kk] = vv
        r["env"] = overrides
        results.append(r)
        write_results(out, results)
    print(f"\n{'config':10s} {'sps_med':>8s} {'sps_p10':>8s} {'wall_sps':>8s} {'coll_med':>8s} {'idle/wkr':>8s} {'miss':>6s} {'rows':>5s}")
    for r in sorted(results, key=lambda r: -(r.get("sps_median") or 0)):
        print(f"n{r['n_envs']}k{r['k_standby']:<3d}{','.join(r.get('env',{}).values())[:22]:22s} {r.get('sps_median',0):8.1f} {r.get('sps_p10',0):8.1f} {r.get('wall_sps',0):8.1f} "
              f"{r.get('collection_sps_median',0):8.1f} {r.get('barrier_idle_per_worker_s',0):8.2f} {(r.get('pool_miss_rate') or 0):6.2f} {r['rows_measured']:5d}")
    return 0 if all(r.get("valid_point", False) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
