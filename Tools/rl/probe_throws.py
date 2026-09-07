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
            for _ in range(a.max_episode_steps):
                o = torch.as_tensor(norm.normalize(obs), dtype=torch.float32).unsqueeze(0)
                act = deterministic_action(policy, o)
                obs, _, done, info = env.step(np.asarray(act, dtype=np.float32).ravel())
                kos += int(info.get("hostile_kos_this_step", 0) or 0)
                for ent in layout.all_entities(obs[-layout.total_floats:]):
                    if ent["valid"] and ent.get("has_weapon", 0) > 0.5:
                        armed_seen += 1
                if done:
                    break
            wins += int(kos >= max(1, a.opponents))
            print(f"  ep{ep}: kos={kos} armed-entity-observations={armed_seen}", flush=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
