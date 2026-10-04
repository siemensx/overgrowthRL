#!/usr/bin/env python3
"""OGRL-20261004-019: copy a trained policy into a FRESH network (online distillation / DAgger).

Why: the run21->run27->run31->run35 actor has ~95% of its tanh activations saturated and its weights
have stopped moving (OGRL-20261004-018) -- it has lost plasticity. A fresh network that reproduces
its decisions keeps the skill and gets back the ability to learn. The student can use LayerNorm before
each tanh (--layer-norm) so it does not saturate again.

How: the teacher's normalisers are copied and frozen, so both networks see identical inputs. Each
iteration collects --n-steps decisions per engine in the real scenario mix (ScenarioSampler restored
from the teacher's checkpoint, incl. move-school stage / ground-only fights), acting with the TEACHER
for the first --teacher-acts-iters iterations and with the STUDENT afterwards (DAgger: the student also
learns the states its own mistakes lead to). Every state is labelled with the teacher's continuous
mean (pre-tanh), button logits and value; the student minimises
    MSE(mean) + BCE(button probs vs teacher probs) + value_coef * MSE(value)
over a replay of recent labelled states. continuous_log_std is copied from the teacher.

Output: a checkpoint in train_vec's format (resumable with --resume-from) carrying the teacher's
global_step, normalisers and curriculum position, plus distill_metrics.jsonl next to it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Tools/rl
sys.path.insert(0, str(Path(__file__).resolve().parent))          # Tools/rl/ppo
from vec_env import VecOvergrowthEnv  # noqa: E402
from obs_schema import DEFAULT_LAYOUT  # noqa: E402
from curriculum import MOVE_SCHOOL_STAGES, ScenarioSampler  # noqa: E402
from reward import win_v2_reward_config  # noqa: E402
from policy import ActorCritic, CONTINUOUS_DIM  # noqa: E402
from normalize import ObservationNormalizer, RewardNormalizer  # noqa: E402
from train import _save_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True)
    p.add_argument("--out", required=True, help="student checkpoint path (train_vec format)")
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    p.add_argument("--levels", required=True)
    p.add_argument("--n-envs", type=int, default=20)
    p.add_argument("--k-standby", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--shm-prefix", default="/ogrl_dst")
    p.add_argument("--seed", type=int, default=36)
    p.add_argument("--engine-config-line", action="append", default=[])
    p.add_argument("--move-school", action="store_true", help="sample the move-school scenario mix")
    p.add_argument("--opponents-cap", type=int, default=3)
    p.add_argument("--layer-norm", action="store_true")
    p.add_argument("--total-steps", type=int, default=6_000_000)
    p.add_argument("--teacher-acts-iters", type=int, default=60)
    p.add_argument("--replay", type=int, default=150_000, help="ring buffer of labelled states (~8.3 KB each)")
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--grad-steps", type=int, default=40, help="per iteration")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--act-period", type=int, default=4)
    p.add_argument("--hard-reset-every", type=int, default=20)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--max-wall-hours", type=float, default=0.0, help="exit 75 for an engine recycle; resumes")
    return p.parse_args()


@torch.no_grad()
def heads(policy: ActorCritic, x: torch.Tensor):
    f = policy._features(x)
    a = policy.actor_trunk(f)
    mean = policy.continuous_mean(a)
    logits = policy.discrete_logits(a)
    value = policy.critic_out(policy.critic_trunk(f)).squeeze(-1)
    return mean, logits, value


def student_heads(policy: ActorCritic, x: torch.Tensor):
    f = policy._features(x)
    a = policy.actor_trunk(f)
    return policy.continuous_mean(a), policy.discrete_logits(a), policy.critic_out(policy.critic_trunk(f)).squeeze(-1)


@torch.no_grad()
def sample_action(mean, logits, log_std):
    raw = mean + torch.randn_like(mean) * log_std.exp()
    return torch.cat([torch.tanh(raw), torch.bernoulli(torch.sigmoid(logits))], dim=-1)


def main() -> int:
    a = parse_args()
    torch.set_num_threads(a.torch_threads)
    t_start = time.time()
    layout = DEFAULT_LAYOUT
    tck = torch.load(a.teacher, map_location="cpu", weights_only=False)
    fs = int(tck.get("frame_stack", 4))
    teacher = ActorCritic(layout, frame_stack=fs)
    teacher.load_state_dict(tck["policy"])
    teacher.eval()
    nrm = ObservationNormalizer(layout, frame_stack=fs)
    nrm.load_state_dict(tck["obs_normalizer"])
    out = Path(a.out)
    state_path = out.with_suffix(".distill_state.pt")

    student = ActorCritic(layout, frame_stack=fs, layer_norm=a.layer_norm)
    with torch.no_grad():
        student.continuous_log_std.copy_(teacher.continuous_log_std)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr, eps=1e-5)
    it0, steps0 = 0, 0
    if state_path.exists():  # resume after an engine recycle
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        student.load_state_dict(st["student"])
        opt = torch.optim.Adam(student.parameters(), lr=a.lr, eps=1e-5)
        opt.load_state_dict(st["opt"])
        it0, steps0 = st["iteration"], st["steps"]
        print(f"[distill] resumed at iteration {it0}, {steps0:,} steps", flush=True)

    sampler = ScenarioSampler(d_max_start=1.0, d_max_cap=1.0, d_min=1.0, opponents=a.opponents_cap,
                              opponents_cap=a.opponents_cap, opp_sampling="learnability", rng_seed=a.seed + it0,
                              move_school_stages=MOVE_SCHOOL_STAGES if a.move_school else ())
    sampler.load_curriculum_state(tck.get("curriculum"))
    levels = [s.strip() for s in a.levels.split(",") if s.strip()]
    venv = VecOvergrowthEnv(n_envs=a.n_envs, repo_root=a.repo_root, level=levels, shm_prefix=a.shm_prefix,
                            base_seed=a.seed * 1000 + it0, layout=layout, reward_config=win_v2_reward_config(),
                            frame_stack=fs, max_episode_steps=a.max_episode_steps, k_standby=a.k_standby,
                            act_period=a.act_period, soft_reset=True, hard_reset_every=a.hard_reset_every,
                            scenario_fn=sampler.sample_episode, engine_config_lines=list(a.engine_config_line))
    mlog = open(out.with_name(out.stem + "_distill_metrics.jsonl"), "a")
    D = layout.total_floats * fs
    RX = np.zeros((a.replay, D), dtype=np.float32)
    RY = np.zeros((a.replay, CONTINUOUS_DIM + 6 + 1), dtype=np.float32)
    rpos, rfill = 0, 0
    recycle = False
    it, steps = it0, steps0
    try:
        obs = venv.reset()
        while steps < a.total_steps:
            if a.max_wall_hours > 0 and time.time() - t_start > a.max_wall_hours * 3600:
                recycle = True
                break
            student.eval()
            teacher_acts = it < a.teacher_acts_iters
            outcomes = {"won": 0, "lost": 0, "timeout": 0}
            agree = []
            t0 = time.time()
            for _ in range(a.n_steps):
                x = torch.as_tensor(nrm.normalize(obs, update=False), dtype=torch.float32)
                tm, tl, tv = heads(teacher, x)
                sm, sl, _sv = heads(student, x)
                agree.append(float(((tl > 0) == (sl > 0)).float().mean()))
                if teacher_acts:
                    act = sample_action(tm, tl, teacher.continuous_log_std)
                else:
                    act = sample_action(sm, sl, student.continuous_log_std)
                n = x.shape[0]
                ids = (rpos + np.arange(n)) % a.replay
                RX[ids] = x.numpy()
                RY[ids] = np.concatenate([tm.numpy(), tl.numpy(), tv.numpy()[:, None]], axis=1)
                rpos = (rpos + n) % a.replay
                rfill = min(a.replay, rfill + n)
                obs, _r, term, trunc, infos = venv.step(act.numpy())
                for i in np.where(np.logical_or(term, trunc))[0]:
                    inf = infos[i]
                    won = bool(inf.get("won"))
                    to = bool(inf.get("timed_out", not term[i]))
                    outcomes["won" if won else ("timeout" if to else "lost")] += 1
                    sc = inf.get("scenario") or {}
                    if sc.get("ground_only"):
                        sampler.record_ground_outcome(sc.get("opponents", 1) or 1, won, sc.get("difficulty"))
                    else:
                        sampler.record_opponent_outcome(sc.get("opponents", 1) or 1, won)
            steps += a.n_steps * a.n_envs
            collect_s = time.time() - t0

            student.train()
            losses = []
            t1 = time.time()
            for _ in range(a.grad_steps):
                idx = np.random.randint(0, rfill, size=min(a.minibatch, rfill))
                xb = torch.as_tensor(RX[idx])
                yb = torch.as_tensor(RY[idx])
                m, l, v = student_heads(student, xb)
                # match the stick as PLAYED (tanh of the mean): the teacher's pre-tanh means are huge
                # (part of its saturation) and matching them past tanh's range changes nothing.
                loss_c = 10.0 * F.mse_loss(torch.tanh(m), torch.tanh(yb[:, :CONTINUOUS_DIM]))
                loss_d = F.binary_cross_entropy_with_logits(l, torch.sigmoid(yb[:, CONTINUOUS_DIM:CONTINUOUS_DIM + 6]))
                loss_v = F.mse_loss(v, yb[:, -1])
                loss = loss_c + loss_d + a.value_coef * loss_v
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                losses.append((loss_c.item(), loss_d.item(), loss_v.item()))
            it += 1
            lc, ld, lv = (float(np.mean([q[i] for q in losses])) for i in range(3))
            row = {"t": time.time(), "iteration": it, "steps": steps, "actor": "teacher" if teacher_acts else "student",
                   "outcomes": outcomes, "button_mode_agreement": float(np.mean(agree)),
                   "loss_mean": lc, "loss_buttons": ld, "loss_value": lv, "replay": rfill,
                   "collect_seconds": round(collect_s, 1), "train_seconds": round(time.time() - t1, 1),
                   "move_school": sampler.move_school_snapshot()}
            mlog.write(json.dumps(row) + "\n")
            mlog.flush()
            print(f"[distill] it={it} steps={steps:,} actor={row['actor']} agree={row['button_mode_agreement']:.4f} "
                  f"loss mean={lc:.4f} buttons={ld:.4f} value={lv:.4f} outcomes={outcomes} "
                  f"sps={a.n_steps * a.n_envs / max(collect_s, 1e-6):.0f}", flush=True)
            if it % 10 == 0:
                torch.save({"student": student.state_dict(), "opt": opt.state_dict(), "iteration": it, "steps": steps},
                           str(state_path) + ".tmp")
                os.replace(str(state_path) + ".tmp", state_path)
    finally:
        try:
            torch.save({"student": student.state_dict(), "opt": opt.state_dict(), "iteration": it, "steps": steps},
                       str(state_path) + ".tmp")
            os.replace(str(state_path) + ".tmp", state_path)
        except Exception as e:  # noqa: BLE001
            print(f"[distill] state save failed: {e}", flush=True)
        venv.close()

    # Final checkpoint in train_vec format: the teacher's step, normalisers and curriculum position.
    rn = RewardNormalizer(0.997, n_envs=a.n_envs)
    if "reward_normalizer" in tck:
        rn.load_state_dict(tck["reward_normalizer"])
    fresh_opt = torch.optim.Adam(student.parameters(), lr=3e-4, eps=1e-5)
    os.environ.setdefault("OGRL_ALLOW_CHECKPOINT_REGRESSION", "1")
    cur = dict(tck.get("curriculum") or {})
    if cur.get("move_school_stage") is not None:
        cur["move_school_stage_start"] = int(tck["global_step"])  # the student gets a full window in its stage
    _save_checkpoint(str(out), student, fresh_opt, nrm, rn, int(tck["global_step"]), curriculum=cur)
    print(f"[distill] wrote {out} (layer_norm={student.layer_norm}, global_step={int(tck['global_step']):,})", flush=True)
    return 75 if recycle else 0


if __name__ == "__main__":
    sys.exit(main())
