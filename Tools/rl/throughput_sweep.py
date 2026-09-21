#!/usr/bin/env python3
"""End-to-end throughput sweep of the REAL trainer, one config at a time.

concurrency_sweep.py measures the collector ceiling with random actions on
the default scenario. That is not the number that matters for run21: the
live fight is 1v3 at difficulty 1.0, episodes run 600-1200 decisions, and
resets, the PPO update and the standby pool all interact. This launches
train_vec.py itself for a fixed wall time per (n_envs, k_standby) point,
resumed from a real checkpoint into a throwaway checkpoint path, then reads
the run's own metrics.jsonl and reports median steps_per_second_cycle over
the measurement window (first `warmup` seconds discarded: engine startup).

    python Tools/rl/throughput_sweep.py --resume-from <ckpt> --grid 10x2 10x4 12x2 12x4 8x2 14x2

Kills every Overgrowth.exe between points, so run it on an otherwise idle
trainer. Writes one JSON with every point's numbers and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics as st
import subprocess
import sys
import time
from pathlib import Path


def kill_engines() -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/IM", "Overgrowth.exe"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "write-dir.*env-ogrl_sw"], capture_output=True)


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
    if run_dir.exists():
        shutil.rmtree(run_dir)
    ckpt = repo / "Tools" / "rl" / "ppo" / "checkpoints" / f"{run_id}.pt"
    if ckpt.exists():
        ckpt.unlink()
    cmd = [sys.executable, "-u", str(repo / "Tools" / "rl" / "ppo" / "train_vec.py"),
           "--repo-root", str(repo), "--levels", args.levels,
           "--shm-prefix", shm_prefix, "--n-envs", str(n_envs), "--k-standby", str(k_standby),
           "--seed", "7", "--checkpoint-path", str(ckpt), "--resume-from", args.resume_from,
           "--run-id", run_id, "--total-timesteps", "4000000000",
           "--n-steps", "256", "--n-epochs", "1", "--minibatch-size", "128",
           "--entropy-coef", "0.003", "--entropy-coef-final", "0.003", "--entropy-anneal-steps", "1000000",
           "--learning-rate", "0.0003", "--target-kl", "0.02", "--max-episode-steps", "1200",
           "--frame-stack", "4", "--act-period", "4", "--soft-reset", "--hard-reset-every", "50",
           "--d-max-start", "1.0", "--d-max-cap", "1.0", "--d-step", "0.1", "--d-min", "1.0",
           "--opponents-cap", "3", "--opp-keep-solo", "0.0", "--armed-stage", "0",
           "--gate-eval-episodes", "30", "--gate-min-step-gap", "999999999999",
           "--collection-torch-threads", str(args.collection_threads),
           "--update-torch-threads", str(args.update_threads),
           "--torch-interop-threads", str(args.interop_threads),
           "--no-tapes", "--no-native-capture"] + args.extra
    env = dict(os.environ, OGRL_ALLOW_NENVS_CHANGE="1")
    log = run_dir.parent / f"{run_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {run_id}: {args.warmup + args.measure:.0f}s ===", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo), stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            proc.wait(timeout=args.warmup + args.measure)
            exited_early = True
        except subprocess.TimeoutExpired:
            exited_early = False
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    kill_engines()
    time.sleep(5)
    metrics = run_dir / "metrics.jsonl"
    rows = []
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    win = [r for r in rows if r["t"] - t0 >= args.warmup]
    def col(k):
        return [r["perf"][k] for r in win if r.get("perf", {}).get(k) is not None]
    sps = col("steps_per_second_cycle")
    out = {"run_id": run_id, "n_envs": n_envs, "k_standby": k_standby, "rows_total": len(rows), "rows_measured": len(win),
           "exited_early": exited_early, "wall_seconds": time.time() - t0}
    if sps:
        out.update({
            "sps_median": st.median(sps), "sps_mean": st.mean(sps), "sps_p10": sorted(sps)[len(sps) // 10],
            "collection_sps_median": st.median(col("steps_per_second_collection") or [0]),
            "barrier_idle_per_worker_s": st.mean(col("barrier_idle_seconds") or [0]) / n_envs,
            "cycle_s": st.median(col("cycle_seconds") or [0]),
            "pool_miss_rate": (sum(col("pool_misses")) / max(1, sum(col("pool_misses")) + sum(col("pool_hits")))) if col("pool_hits") else None,
            "steps_measured": (win[-1]["global_step"] - win[0]["global_step"]) if len(win) > 1 else 0,
            "wall_sps": ((win[-1]["global_step"] - win[0]["global_step"]) / (win[-1]["t"] - win[0]["t"])) if len(win) > 1 else 0,
        })
    print(json.dumps(out), flush=True)
    for p in (ckpt,):
        if p.exists():
            p.unlink()
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
    ap.add_argument("--collection-threads", type=int, default=2,
                    help="PyTorch intra-op threads during rollout inference")
    ap.add_argument("--update-threads", type=int, default=4)
    ap.add_argument("--interop-threads", type=int, default=1,
                    help="PyTorch inter-op threads")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    kill_engines()
    results = []
    for g in args.grid:
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
        out = Path(args.out) if args.out else Path(args.repo_root) / "Tools" / "rl" / "runs" / f"throughput_sweep_{args.tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))
    print(f"\n{'config':10s} {'sps_med':>8s} {'sps_p10':>8s} {'wall_sps':>8s} {'coll_med':>8s} {'idle/wkr':>8s} {'miss':>6s} {'rows':>5s}")
    for r in sorted(results, key=lambda r: -(r.get("sps_median") or 0)):
        print(f"n{r['n_envs']}k{r['k_standby']:<3d}{','.join(r.get('env',{}).values())[:22]:22s} {r.get('sps_median',0):8.1f} {r.get('sps_p10',0):8.1f} {r.get('wall_sps',0):8.1f} "
              f"{r.get('collection_sps_median',0):8.1f} {r.get('barrier_idle_per_worker_s',0):8.2f} {(r.get('pool_miss_rate') or 0):6.2f} {r['rows_measured']:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
