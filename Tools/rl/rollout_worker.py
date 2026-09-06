#!/usr/bin/env python3
"""Remote rollout worker: owns engines, owns no learning.

Connects to a learner, and for each update receives the current policy plus
observation-normaliser state, collects exactly n_steps transitions for its own
envs, and returns the raw arrays. It never computes a gradient and never
normalises rewards -- the learner does both, so normalisation sees one stream.

Run on the FAST machine and give it env slots proportional to its speed; see
remote_rollout.py for why equal splitting is worse than not distributing at all.

    python3 Tools/rl/rollout_worker.py --learner 192.168.99.33:5599 --n-envs 12
"""
from __future__ import annotations
import argparse, socket, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ppo"))

from remote_rollout import send_msg, recv_msg, configure_socket  # noqa: E402
from vec_env import VecOvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT  # noqa: E402
from curriculum import ScenarioSampler  # noqa: E402
from normalize import ObservationNormalizer, RewardNormalizer  # noqa: E402
from policy import ActorCritic  # noqa: E402
from env import ACTION_DIM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", required=True, help="HOST:PORT of the learner")
    ap.add_argument("--n-envs", type=int, required=True)
    ap.add_argument("--k-standby", type=int, default=2)
    ap.add_argument("--repo-root", default=str(HERE.parent.parent))
    ap.add_argument("--shm-prefix", default="/ogrl_rw")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    host, port = a.learner.split(":")
    sock = socket.create_connection((host, int(port)), timeout=120)
    configure_socket(sock)
    send_msg(sock, {"type": "hello", "n_envs": a.n_envs})
    cfg = recv_msg(sock)
    if cfg.get("type") != "config":
        raise RuntimeError(f"expected config, got {cfg.get('type')}")
    print(f"joined learner: n_steps={cfg['n_steps']} levels={len(cfg['levels'])} "
          f"act_period={cfg['act_period']} frame_stack={cfg['frame_stack']}")

    layout = DEFAULT_LAYOUT
    sampler = ScenarioSampler(**cfg["sampler_kwargs"])
    env = VecOvergrowthEnv(
        n_envs=a.n_envs, repo_root=a.repo_root, level=cfg["levels"], shm_prefix=a.shm_prefix,
        base_seed=a.seed, layout=layout, frame_stack=cfg["frame_stack"],
        max_episode_steps=cfg["max_episode_steps"], act_period=cfg["act_period"],
        k_standby=a.k_standby, soft_reset=True, hard_reset_every=cfg["hard_reset_every"],
        scenario_fn=sampler.sample_episode)
    device = torch.device(a.device)
    policy = ActorCritic(layout, frame_stack=cfg["frame_stack"]).to(device)
    obs_norm = ObservationNormalizer(layout, frame_stack=cfg["frame_stack"])
    # running_return is a PER-ENV discounted accumulator, so this worker owning
    # the state for its own envs is correct -- those envs exist only here. The
    # shared part is the reward SCALE (rms mean/var/count), which the learner
    # broadcasts each update, so both machines normalise on one common scale.
    rew_norm = RewardNormalizer(cfg["gamma"], n_envs=a.n_envs)
    raw_obs = env.reset()
    n_steps = cfg["n_steps"]
    obs_dim = layout.total_floats * cfg["frame_stack"]

    try:
        while True:
            msg = recv_msg(sock)
            if msg.get("type") == "stop":
                print("learner said stop"); break
            if msg.get("type") != "weights":
                raise RuntimeError(f"unexpected message {msg.get('type')}")
            policy.load_state_dict(msg["policy"])
            obs_norm.load_state_dict(msg["obs_normalizer"])
            rn = dict(msg["reward_normalizer"])
            import numpy as _np
            rn["running_return"] = _np.zeros(a.n_envs, dtype=_np.float64) \
                if len(rn.get("running_return", [])) != a.n_envs else rn["running_return"]
            rew_norm.load_state_dict(rn)
            policy.eval()

            t0 = time.monotonic()
            O = np.zeros((n_steps, a.n_envs, obs_dim), dtype=np.float32)
            A = np.zeros((n_steps, a.n_envs, ACTION_DIM), dtype=np.float32)
            LP = np.zeros((n_steps, a.n_envs), dtype=np.float32)
            V = np.zeros((n_steps, a.n_envs), dtype=np.float32)
            R = np.zeros((n_steps, a.n_envs), dtype=np.float32)   # normalised on the learner's scale
            D = np.zeros((n_steps, a.n_envs), dtype=np.float32)
            ep = []
            for t in range(n_steps):
                norm = obs_norm.normalize(raw_obs, update=True)
                with torch.no_grad():
                    tens = torch.as_tensor(norm, dtype=torch.float32, device=device)
                    act, logp, _entropy, val = policy.get_action_and_value(tens)
                act_np = act.cpu().numpy()
                O[t] = norm; A[t] = act_np
                LP[t] = logp.cpu().numpy(); V[t] = val.cpu().numpy()
                raw_next, rew, term, trunc, infos = env.step(act_np)
                stop = np.logical_or(term, trunc)
                R[t] = rew_norm.normalize(np.asarray(rew, dtype=np.float32), stop)
                D[t] = stop.astype(np.float32)
                for i, (te, tr) in enumerate(zip(term, trunc)):
                    if te or tr:
                        sc = infos[i].get("scenario", {}) or {}
                        ep.append({"opponents": sc.get("opponents", 1),
                                   "won": bool(infos[i]["reward_components"].get("opponent_knockout", 0) > 0
                                               and te),
                                   "difficulty": sc.get("difficulty")})
                raw_obs = raw_next
            with torch.no_grad():
                last_v = policy.get_value(
                    torch.as_tensor(obs_norm.normalize(raw_obs, update=False),
                                    dtype=torch.float32, device=device)).cpu().numpy()
            send_msg(sock, {"type": "rollout", "obs": O, "actions": A, "log_probs": LP,
                            "values": V, "rewards": R, "dones": D, "last_values": last_v,
                            "obs_norm": obs_norm.state_dict(), "episodes": ep,
                            "collect_seconds": time.monotonic() - t0})
            print(f"  sent rollout: {n_steps}x{a.n_envs} in {time.monotonic()-t0:.2f}s "
                  f"({n_steps*a.n_envs/(time.monotonic()-t0):.0f} decisions/s)", flush=True)
    finally:
        env.close(); sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
