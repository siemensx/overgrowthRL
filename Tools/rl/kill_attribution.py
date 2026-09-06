#!/usr/bin/env python3
"""How does the agent actually finish opponents?

Win rate says whether it wins. This says HOW -- which is what tells you whether
the policy has a repertoire or one dominant trick, and which of the game's
mechanics it has actually discovered.

Overgrowth does not expose "which move landed" as an observation field, so the
attack is classified the way the game itself chooses one (aschar.as's
UpdateAttackState) -- from the CONTEXT the attack was thrown in -- and paired
with the victim's state at the moment it went down:

  * agent airborne              -> jump kick
  * agent crouched              -> crouch kick / leg sweep
  * agent grounded and moving   -> running strike
  * agent grounded and still    -> standing strike
  * victim already in ragdoll   -> ground finisher (a downed opponent being
                                   hit, e.g. after an over-shoulder throw)

It deliberately does NOT infer throws from the grab button. The policy presses
grab on ~71% of decisions (see action_profile.py), so "grab was pressed shortly
before the kill" is satisfied by chance on essentially every kill and measures
button-mashing, not throws. A real throw signal exists as script state
(`attacking_with_throw` on the thrower, `hit_reaction_thrown` on the victim,
aschar.as:7802/5262) and the engine already has the machinery to read script
globals (rl_observation.cpp's ReadIntGlobal/ReadBoolGlobal, used for
fov_focus) -- but surfacing it needs an engine change, so it is not guessed at
here.

What IS reported without guessing: how the victim came to be on the ground in
the first place, classified from the agent's context at the moment the victim
entered ragdoll. A knockdown thrown while the agent is grounded and adjacent is
a very different mechanic from one delivered in mid-air.

Knockouts are read from the engine's own `hostile_kos_this_step` reward
component -- the same signal the win condition uses. An earlier version watched
for a visible entity transitioning awake -> unconscious, which silently found
nothing: the observation is FOV-gated (rl_observation.cpp requires
local_dir.z() > 0), so a victim is frequently not visible at the instant it
goes down. The victim's state is therefore taken from the last time it WAS
seen, within a short window before the knockout.
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
from collections import Counter, deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ppo"))

from env import OvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT as L  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402

DT = 4 / 120.0
LOOKBACK = 4          # decisions of context kept before a knockout (~0.13 s)
GRAB_WINDOW = 12      # decisions to look back for a throw setup (~0.4 s)


def _agent_context(raw, act, mv):
    grounded = raw[L.GROUNDED] > 0.5
    crouch = act[3] > 0.5
    if not grounded:
        return "jump kick (airborne)"
    if crouch:
        return "crouch kick / sweep"
    return "running strike" if mv > 0.5 else "standing strike"


def _victim_state(e):
    names = ["movement", "ground", "attack", "hit_reaction", "ragdoll"]
    st_oh = list(e["state"])
    return names[max(range(len(st_oh)), key=lambda i: st_oh[i])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="Tools/rl/ppo/checkpoints/run21_mac.pt")
    ap.add_argument("--level", default="arenas/t_train_101.xml")
    ap.add_argument("--opponents", type=int, default=1)
    ap.add_argument("--difficulty", type=float, default=0.6)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--shm-name", default="/ogrl_ka")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    fs = int(ck.get("frame_stack", 4))
    pol = ActorCritic(L, frame_stack=fs); pol.load_state_dict(ck["policy"]); pol.eval()
    nrm = ObservationNormalizer(L, frame_stack=fs); nrm.load_state_dict(ck["obs_normalizer"])

    env = OvergrowthEnv(repo_root=os.getcwd(), level=a.level, shm_name=a.shm_name,
                        seed=a.seed, act_period=4, frame_stack=fs)
    env.reset(seed=a.seed, difficulty=a.difficulty, opponents=a.opponents)
    obs = env.reset(seed=a.seed + 1, difficulty=a.difficulty, opponents=a.opponents)

    ctx_hist = deque(maxlen=LOOKBACK)
    grab_hist = deque(maxlen=GRAB_WINDOW)
    last_seen = deque(maxlen=LOOKBACK)   # most recent state of a visible hostile
    prev_ragdoll = {}                    # entity id -> was in ragdoll last seen
    knockdown_ctx = Counter()
    how = Counter(); victim_was = Counter(); after_throw = 0
    kills = 0; ep = 0; steps = 0; ttk = []
    ep_first_kill = None

    while ep < a.episodes:
        with torch.no_grad():
            x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32).unsqueeze(0)
            act = pol.get_action_and_value(x)[0].squeeze(0).numpy()
        raw_prev = env._prev_values
        obs, _r, done, info = env.step(act)
        steps += 1
        raw = env._prev_values
        if raw is None:
            continue
        raw = np.asarray(raw, dtype=np.float32)
        mv = float(np.hypot(act[0], act[1]))
        ctx_hist.append(_agent_context(np.asarray(raw_prev, dtype=np.float32) if raw_prev is not None else raw,
                                       act, mv))
        grab_hist.append(act[5] > 0.5)          # grab == the throw button

        # Remember the most recent state of any visible hostile, so a victim
        # that has rotated out of the FOV cone can still be characterised.
        for slot in range(L.max_visible_entities):
            e = L.entity_field(raw, slot)
            if e["valid"] and not e["is_ally"] and e["knocked_out"][0] > 0.5:
                vs = _victim_state(e)
                last_seen.append(vs)
                # A knockdown is a transition INTO ragdoll, attributed to what
                # the agent was doing at that moment. No button presses involved.
                was = prev_ragdoll.get(e["id"])
                if was is False and vs == "ragdoll":
                    knockdown_ctx[ctx_hist[-1] if ctx_hist else "?"] += 1
                prev_ragdoll[e["id"]] = (vs == "ragdoll")

        rc = (info or {}).get("reward_components", {}) or {}
        n_ko = rc.get("hostile_kos_this_step")
        n_ko = int(round(n_ko)) if n_ko is not None else (1 if rc.get("opponent_knockout", 0) > 0 else 0)
        for _ in range(n_ko):
            kills += 1
            how[ctx_hist[-1] if ctx_hist else "?"] += 1
            victim_was[last_seen[-1] if last_seen else "not visible"] += 1
            if any(grab_hist):
                after_throw += 1
            if ep_first_kill is None:
                ep_first_kill = steps

        if done:
            if ep_first_kill:
                ttk.append(ep_first_kill)
            ep += 1; steps = 0; ep_first_kill = None
            last_seen.clear(); ctx_hist.clear(); grab_hist.clear(); prev_ragdoll.clear()
            obs = env.reset(seed=a.seed + 100 + ep, difficulty=a.difficulty, opponents=a.opponents)
    env.close()

    print(f"checkpoint step={ck['global_step']:,}  level={a.level}  "
          f"opponents={a.opponents}  difficulty={a.difficulty}  episodes={a.episodes}")
    print(f"attributed knockouts: {kills}\n")
    if not kills:
        print("no attributed knockouts -- raise --episodes or lower --difficulty")
        return 0
    print("HOW the finishing blow was thrown (agent context):")
    for k, v in how.most_common():
        print(f"  {k:26} {v:5d}  {100.0*v/kills:5.1f}%")
    print("\nWHAT STATE the victim was in when it went down:")
    for k, v in victim_was.most_common():
        tag = "  <- downed opponent being finished" if k == "ragdoll" else ""
        print(f"  {k:26} {v:5d}  {100.0*v/kills:5.1f}%{tag}")
    tot_kd = sum(knockdown_ctx.values())
    if tot_kd:
        print(f"\nHOW opponents were KNOCKED DOWN in the first place ({tot_kd} knockdowns):")
        for k, v in knockdown_ctx.most_common():
            print(f"  {k:26} {v:5d}  {100.0*v/tot_kd:5.1f}%")
    print("\nNOTE: throw detection is deliberately omitted. grab is pressed on ~71% of")
    print("decisions, so any 'grab before the kill' statistic measures mashing, not throws.")
    print("The real signal is script state (attacking_with_throw / hit_reaction_thrown)")
    print("and needs an engine change to surface.")
    if ttk:
        print(f"time to first knockout: median {st.median(ttk)*DT:.1f}s  "
              f"p10 {np.percentile(ttk,10)*DT:.1f}s  p90 {np.percentile(ttk,90)*DT:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
