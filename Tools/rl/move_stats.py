#!/usr/bin/env python3
"""Which moves does the policy actually throw, from the engine's OWN resolution.

Earlier attempts at this inferred the move from the agent's context (airborne /
crouched / moving) and the victim's state at the instant of the knockout. Both
were wrong in ways that mattered:

  * "grab was pressed shortly before the kill" measures mashing -- grab is
    pressed on ~71% of decisions, so it is satisfied by chance.
  * "the victim was in ragdoll when it died" is close to tautological: a
    knockout PUTS a character into ragdoll, so reading the state at the moment
    the KO registers reports the consequence of the blow, not its precondition.

`aschar.as::GetAttackPath` is the single place where every attack resolves to a
concrete move file, and it already receives `ragdoll_enemy` and `ducking_enemy`
as inputs -- the game's own answer to "am I hitting someone who is down?",
decided BEFORE the blow lands. With `rl_log_attacks` set, each resolution is
logged; this parses that.

Nothing is inferred here. The move name is the attack XML the engine chose.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ppo"))

from env import OvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402

RLATK = re.compile(
    r"RLATK id=(\d+) kind=(\S+) path=(\S+) ragdoll=(\d) ducking=(\d) dist=([\d.eE+-]+)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="Tools/rl/ppo/checkpoints/run21_mac.pt")
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=1)
    ap.add_argument("--difficulty", type=float, default=0.6)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--shm-name", default=None)
    ap.add_argument("--out", default="Tools/rl/runs/run21_mac/eval/move_stats.json",
                    help="written for the dashboard's move-distribution panel")
    a = ap.parse_args()
    if a.shm_name is None:
        a.shm_name = f"/ogrl_ms{os.getpid() % 100000:05d}"

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    fs = int(ck.get("frame_stack", 4))
    pol = ActorCritic(L, frame_stack=fs); pol.load_state_dict(ck["policy"]); pol.eval()
    nrm = ObservationNormalizer(L, frame_stack=fs); nrm.load_state_dict(ck["obs_normalizer"])

    env = OvergrowthEnv(repo_root=os.getcwd(), level=a.level, shm_name=a.shm_name,
                        seed=a.seed, act_period=4, frame_stack=fs, log_attacks=True)
    log_path = env._write_dir.parent / (env._write_dir.name + ".log")
    env.reset(seed=a.seed, difficulty=a.difficulty, opponents=a.opponents)
    obs = env.reset(seed=a.seed + 1, difficulty=a.difficulty, opponents=a.opponents)

    self_id = None
    ep = 0
    while ep < a.episodes:
        with torch.no_grad():
            act = pol.get_action_and_value(
                torch.as_tensor(nrm.normalize(obs, update=False),
                                dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()
        obs, _r, done, _i = env.step(act)
        if self_id is None and env._prev_values is not None:
            self_id = int(np.asarray(env._prev_values, dtype=np.float32)[L.SELF_ID])
        if done:
            ep += 1
            obs = env.reset(seed=a.seed + 100 + ep, difficulty=a.difficulty, opponents=a.opponents)
    tmp = f"/tmp/move_stats_{os.getpid()}.log"
    shutil.copy(log_path, tmp)
    env.close()

    agent = Counter(); agent_ragdoll = Counter(); other = Counter()
    n_agent = 0
    for line in open(tmp, errors="replace"):
        m = RLATK.search(line)
        if not m:
            continue
        cid, kind, path, ragdoll, ducking, _dist = m.groups()
        move = os.path.basename(path).replace(".xml", "")
        if self_id is not None and int(cid) == self_id:
            agent[move] += 1
            n_agent += 1
            if ragdoll == "1":
                agent_ragdoll[move] += 1
        else:
            other[move] += 1

    print(f"checkpoint step={ck['global_step']:,}  opponents={a.opponents}  "
          f"difficulty={a.difficulty}  episodes={a.episodes}  agent id={self_id}")
    print(f"attacks thrown by the AGENT: {n_agent}\n")
    if not n_agent:
        print("no RLATK lines attributed to the agent -- is rl_log_attacks reaching the engine?")
        return 1
    print(f"{'move (engine-resolved attack file)':38} {'count':>7} {'share':>7} {'vs downed':>10}")
    for mv, c in agent.most_common():
        dr = agent_ragdoll[mv]
        print(f"  {mv:36} {c:>7} {100.0*c/n_agent:>6.1f}% {dr:>6} ({100.0*dr/c:>4.0f}%)")
    tot_rag = sum(agent_ragdoll.values())
    print(f"\nattacks aimed at an opponent ALREADY DOWN: {tot_rag}/{n_agent} = "
          f"{100.0*tot_rag/n_agent:.1f}%")
    print("  (ragdoll_enemy as the engine evaluated it BEFORE the blow -- not the")
    print("   victim's state afterwards, which a knockout sets by definition)")
    if other:
        print(f"\nfor reference, opponents' own attacks: {sum(other.values())}")
        for mv, c in other.most_common(5):
            print(f"  {mv:36} {c:>7} {100.0*c/sum(other.values()):>6.1f}%")

    if a.out:
        import json, time
        from pathlib import Path
        payload = {
            "t": time.time(),
            "global_step": int(ck["global_step"]),
            "level": a.level, "opponents": a.opponents, "difficulty": a.difficulty,
            "episodes": a.episodes,
            "agent_attacks": n_agent,
            "agent_moves": dict(agent),
            "agent_moves_vs_downed": dict(agent_ragdoll),
            "vs_downed_share": (tot_rag / n_agent) if n_agent else None,
            "opponent_moves": dict(other),
        }
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        prev = []
        if out.exists():
            try: prev = json.loads(out.read_text()).get("history", [])
            except Exception: prev = []
        prev.append(payload)
        out.write_text(json.dumps({"history": prev[-40:]}, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
