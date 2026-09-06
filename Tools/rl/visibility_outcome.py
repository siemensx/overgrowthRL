#!/usr/bin/env python3
"""Does poor visibility actually predict LOSING, or is it merely present?

measure_visibility.py established that the observation is severely partial: at
three opponents the policy sees all three on only ~6% of decisions (mean 1.54 of
3), and contiguous blind intervals (median 0.43 s) are several times longer than
the 0.133 s the frame stack can remember. It did NOT establish that this is what
caps multi-opponent performance -- its damage-event samples were n=9..33, and
awkwardly, 1v1 has the WORST visibility (38% of decisions see nothing, blind
median 1.80 s) and the BEST win rate.

So test the causal claim directly and with enough samples: run many episodes,
record per-episode mean visible-hostile fraction and the outcome, and compare
visibility in wins against losses. If visibility does not separate them, a
richer observation (remembered entities) or a recurrent core is not the thing
holding multi-opponent back, whatever else is true about the representation.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ppo"))

from env import OvergrowthEnv, ACTION_DIM  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402


def _visible_hostiles(raw: np.ndarray) -> int:
    n = 0
    for slot in range(L.max_visible_entities):
        e = L.entity_field(raw, slot)
        if e["valid"] and not e["is_ally"] and e["knocked_out"][0] > 0.5:
            n += 1
    return n


def _welch(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t and a normal-approx two-sided p. Unequal variances, unequal n."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="Tools/rl/ppo/checkpoints/run21_mac.pt")
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--difficulty", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--shm-name", default="/ogrl_vo")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    fs = int(ck.get("frame_stack", 4))
    pol = ActorCritic(L, frame_stack=fs); pol.load_state_dict(ck["policy"]); pol.eval()
    nrm = ObservationNormalizer(L, frame_stack=fs); nrm.load_state_dict(ck["obs_normalizer"])

    env = OvergrowthEnv(repo_root=os.getcwd(), level=a.level, shm_name=a.shm_name,
                        seed=a.seed, act_period=4, frame_stack=fs)
    env.reset(seed=a.seed, difficulty=a.difficulty, opponents=a.opponents)
    obs = env.reset(seed=a.seed + 1, difficulty=a.difficulty, opponents=a.opponents)

    # Episode-mean visibility is CONFOUNDED by episode length: opponents spawn
    # in front of the agent, so the opening decisions are high-visibility, and a
    # short episode is dominated by them. A first pass showed losses "seeing
    # more" (t=-3.82) purely because losses are less than half as long. Measure
    # a fixed early window as well, which is length-matched by construction.
    WINDOW = 200
    wins_vis, loss_vis, wins_len, loss_len = [], [], [], []
    wins_win, loss_win = [], []
    vis_win_sum = vis_win_steps = 0
    vis_sum = steps = 0
    kos = 0
    ep = 0
    while ep < a.episodes:
        with torch.no_grad():
            x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32).unsqueeze(0)
            act, _, _, _ = pol.get_action_and_value(x)
        obs, _r, done, info = env.step(act.squeeze(0).numpy())
        raw = env._prev_values
        if raw is not None:
            v = _visible_hostiles(np.asarray(raw, dtype=np.float32))
            vis_sum += v
            steps += 1
            if steps <= WINDOW:
                vis_win_sum += v
                vis_win_steps += 1
        rc = (info or {}).get("reward_components", {}) or {}
        k = rc.get("hostile_kos_this_step")
        kos += int(round(k)) if k is not None else (1 if rc.get("opponent_knockout", 0) > 0 else 0)
        if done:
            frac = (vis_sum / max(1, steps)) / a.opponents
            won = kos >= a.opponents
            (wins_vis if won else loss_vis).append(frac)
            (wins_len if won else loss_len).append(steps)
            if vis_win_steps >= WINDOW:      # only length-matched episodes
                wfrac = (vis_win_sum / vis_win_steps) / a.opponents
                (wins_win if won else loss_win).append(wfrac)
            ep += 1
            vis_sum = steps = kos = 0
            vis_win_sum = vis_win_steps = 0
            obs = env.reset(seed=a.seed + 100 + ep, difficulty=a.difficulty, opponents=a.opponents)
    env.close()

    nw, nl = len(wins_vis), len(loss_vis)
    print(f"checkpoint step={ck['global_step']:,}  level={a.level}  opponents={a.opponents}  "
          f"difficulty={a.difficulty}  episodes={a.episodes}")
    print(f"win rate: {nw}/{a.episodes} = {nw / a.episodes:.3f}\n")
    if nw == 0 or nl == 0:
        print("all episodes had the same outcome -- cannot separate; raise --episodes or change difficulty")
        return 0
    t, p = _welch(wins_vis, loss_vis)
    print(f"WHOLE-EPISODE mean visible fraction (CONFOUNDED by length -- see below):")
    print(f"  wins   n={nw:3d}  mean={st.mean(wins_vis):.3f}  sd={st.pstdev(wins_vis):.3f}")
    print(f"  losses n={nl:3d}  mean={st.mean(loss_vis):.3f}  sd={st.pstdev(loss_vis):.3f}")
    print(f"  Welch t={t:+.2f}  p~{p:.3f}")
    print(f"  episode length: wins {st.mean(wins_len):.0f} decisions, losses {st.mean(loss_len):.0f}")
    print(f"  -> losses are {st.mean(wins_len)/max(1e-9,st.mean(loss_len)):.1f}x shorter, and opponents")
    print(f"     SPAWN IN FRONT, so a short episode is weighted toward its high-visibility opening.")
    nww, nlw = len(wins_win), len(loss_win)
    print(f"\nLENGTH-MATCHED: mean visible fraction over the FIRST {200} decisions only")
    print(f"  (episodes shorter than that are excluded, so both groups cover the same phase)")
    if nww >= 2 and nlw >= 2:
        t2, p2 = _welch(wins_win, loss_win)
        print(f"  wins   n={nww:3d}  mean={st.mean(wins_win):.3f}  sd={st.pstdev(wins_win):.3f}")
        print(f"  losses n={nlw:3d}  mean={st.mean(loss_win):.3f}  sd={st.pstdev(loss_win):.3f}")
        print(f"  Welch t={t2:+.2f}  p~{p2:.3f}")
        print()
        if p2 > 0.05:
            print("  => visibility does NOT separate wins from losses once length is controlled.")
            print("     The whole-episode effect was the confound. Partial observability is REAL")
            print("     (the policy sees ~half its opponents) but is not what decides these fights,")
            print("     so remembered entities / recurrence is not the first lever to pull.")
        elif t2 > 0:
            print("  => wins genuinely see more, length-controlled. Partial observability is")
            print("     implicated; remembered entities are justified.")
        else:
            print("  => losses genuinely see more, length-controlled. That is not a memory")
            print("     problem -- investigate whether seeing more means being surrounded.")
    else:
        print(f"  too few length-matched episodes (wins {nww}, losses {nlw}) -- raise --episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
