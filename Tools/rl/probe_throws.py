#!/usr/bin/env python3
"""Do armed bots actually throw weapons at an AIRBORNE agent?

WantsToThrowItem() in enemycontrol.as claims to prioritise an airborne target,
but only for species == _cat, and only through a hardcoded 0.04 throttle. That
claim had never been observed. This counts RLTHROW telemetry lines (aschar.as
HandleThrow) against a live policy, which is the only way to know the Stage B/C
curriculum has anything behind it.

usage: probe_throws.py --checkpoint C [--species 5] [--armed-count 3]
                       [--weapon-type 1] [--throw-aggression 10] [--episodes 6]
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "ppo"))
from obs_schema import ObsLayout
from env import OvergrowthEnv
from policy import ActorCritic
from normalize import ObservationNormalizer
from watch import deterministic_action


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-root", default=str(HERE.parent.parent))
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--difficulty", type=float, default=0.8)
    ap.add_argument("--species", type=int, default=5)          # 5 = cat
    ap.add_argument("--armed-count", type=int, default=3)
    ap.add_argument("--weapon-type", type=int, default=1)      # 1 = knife
    ap.add_argument("--throw-aggression", type=float, default=1.0)
    ap.add_argument("--max-episode-steps", type=int, default=1200)
    ap.add_argument("--shm-name", default="/ogrl_thr")
    a = ap.parse_args()

    layout = ObsLayout()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    policy = ActorCritic(layout, frame_stack=4)
    policy.load_state_dict(ck["policy"]); policy.eval()
    norm = ObservationNormalizer(layout, frame_stack=4)
    norm.load_state_dict(ck["obs_normalizer"])

    env = OvergrowthEnv(repo_root=a.repo_root, level=a.level, shm_name=a.shm_name,
                        seed=1234, layout=layout, frame_stack=4, act_period=4,
                        render=False, log_attacks=True,
                        throw_aggression_launch=a.throw_aggression)
    log = env._write_dir.parent / (env._write_dir.name + ".log")
    wins = 0
    try:
        for ep in range(a.episodes):
            obs = env.reset(difficulty=a.difficulty, opponents=a.opponents,
                            species=a.species, armed_count=a.armed_count,
                            weapon_type=a.weapon_type, throw_aggression=a.throw_aggression)
            kos = 0
            armed_seen = 0
            air_steps = 0
            air_runs = []
            run_len = 0
            for _ in range(a.max_episode_steps):
                o = torch.as_tensor(norm.normalize(obs), dtype=torch.float32).unsqueeze(0)
                act = deterministic_action(policy, o)
                obs, _, done, info = env.step(np.asarray(act, dtype=np.float32).ravel())
                kos += int(info.get("hostile_kos_this_step", 0) or 0)
                frame = obs[-layout.total_floats:]
                on_ground = float(frame[layout.GROUNDED])
                if on_ground is not None:
                    if on_ground < 0.5:
                        air_steps += 1; run_len += 1
                    elif run_len:
                        air_runs.append(run_len); run_len = 0
                for ent in layout.all_entities(frame):
                    if ent["valid"] and ent.get("has_weapon", 0) > 0.5:
                        armed_seen += 1
                if done:
                    break
            wins += int(kos >= max(1, a.opponents))
            if run_len: air_runs.append(run_len)
            mean_air = (sum(air_runs)/len(air_runs)) if air_runs else 0.0
            print(f"  ep{ep}: kos={kos} armed_obs={armed_seen} airborne_steps={air_steps} "
                  f"jumps={len(air_runs)} mean_airborne={mean_air:.1f} steps "
                  f"({mean_air*4/120:.2f}s)", flush=True)
        # Read the engine log BEFORE close(): env.close() rmtree's the
        # write-dir and removes its sibling .log, so reading afterwards always
        # finds nothing -- which looks exactly like "no throws happened".
        txt = log.read_text(errors="replace") if log.exists() else ""
    finally:
        env.close()

    thr = re.findall(r"RLTHROW .*?target_in_air=(\d+) dist=([\d.]+)", txt)
    atk = len(re.findall(r"RLATK ", txt))
    air = sum(1 for t, _ in thr if t == "1")
    print(f"species={a.species} armed={a.armed_count} weapon={a.weapon_type} "
          f"throw_aggression={a.throw_aggression}  episodes={a.episodes}  agent_wins={wins}")
    print(f"  bot weapon throws committed : {len(thr)}")
    print(f"     ... at an AIRBORNE agent : {air}"
          + (f"  ({100*air/len(thr):.0f}%)" if thr else ""))
    if thr:
        d = [float(x) for _, x in thr]
        print(f"  throw distance mean/min/max : {sum(d)/len(d):.1f} / {min(d):.1f} / {max(d):.1f}")
    print(f"  agent attack events (RLATK) : {atk}")
    dbg = re.findall(r"RLTHROWDBG daa=(\d+) aggr=([\d.]+) prim=(-?\d+) sec=(-?\d+) "
                     r"subgoal=(\d+) avoid=(\d+) dist=([\d.]+)", txt)
    gate = re.findall(r"RLTHROWGATE tether=(\d+) prim=(-?\d+) sec=(-?\d+) knifelayer=(-?\d+) "
                      r"onground=(\d+) flipping=(\d+) target=(-?\d+)", txt)
    if gate:
        n = len(gate)
        print(f"  HandleThrow gate reached     : {n}  (WantsToThrowItem returned TRUE)")
        print(f"     tethered free             : {sum(1 for g in gate if g[0]=='1')}/{n}")
        print(f"     PRIMARY slot has a weapon : {sum(1 for g in gate if g[1]!='-1')}/{n}")
        print(f"     secondary has a weapon    : {sum(1 for g in gate if g[2]!='-1')}/{n}")
        print(f"     knife layer free          : {sum(1 for g in gate if g[3]=='-1')}/{n}")
        print(f"     on ground or flipping     : {sum(1 for g in gate if g[4]=='1' or g[5]=='1')}/{n}")
        print(f"     GetThrowTarget() valid    : {sum(1 for g in gate if g[6]!='-1')}/{n}")
    print(f"  throw-predicate evaluations : {len(dbg)}")
    if dbg:
        daa_on = sum(1 for d in dbg if d[0] == "1")
        armed  = sum(1 for d in dbg if d[2] != "-1" or d[3] != "-1")
        in_goal= sum(1 for d in dbg if d[4] == d[5])
        close  = sum(1 for d in dbg if float(d[6]) <= 8.0)
        both   = sum(1 for d in dbg if d[4] == d[5] and float(d[6]) <= 8.0
                     and (d[2] != "-1" or d[3] != "-1") and d[0] == "1")
        print(f"     g_rl_daa true            : {daa_on}/{len(dbg)}   (aggr={dbg[0][1]})")
        print(f"     holding a weapon         : {armed}/{len(dbg)}")
        print(f"     sub_goal==_avoid_jumpkick: {in_goal}/{len(dbg)}")
        print(f"     within 8 units           : {close}/{len(dbg)}")
        print(f"     ALL FOUR at once         : {both}/{len(dbg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
