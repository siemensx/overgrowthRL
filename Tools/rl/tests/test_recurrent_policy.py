"""OGRL-20261002-013: the recurrent path must re-evaluate stored sequences EXACTLY as rolled out."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "ppo"))
from obs_schema import DEFAULT_LAYOUT  # noqa: E402
from policy import ActorCritic, recurrent_hidden_of  # noqa: E402
from vec_buffer import VecRolloutBuffer  # noqa: E402
import train  # noqa: E402


def _rollout(pol, T=64, E=3, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    D = pol.frame_floats * pol.frame_stack
    buf = VecRolloutBuffer(T, E, D, 8, torch.device("cpu"), hidden_dim=pol.recurrent_hidden)
    h = pol.initial_state(E)
    starts = torch.ones(E)
    for t in range(T):
        obs = torch.randn(E, D) * 0.3
        obs.view(E, pol.frame_stack, pol.frame_floats)[:, :, pol.entities_start] = 1.0  # one valid entity
        with torch.no_grad():
            a, lp, _e, v, raw, h_new = pol.act_recurrent(obs, h, starts)
        done = rng.random(E) < 0.05
        buf.add(obs.numpy(), a.numpy(), lp.numpy(), v.numpy(), rng.normal(size=E).astype(np.float32),
                done.astype(np.float32), raw_cont=raw.numpy(), hidden=h.numpy(), starts=starts.numpy())
        h = h_new
        starts = torch.as_tensor(done, dtype=torch.float32)
    return buf


class TestRecurrent(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.pol = ActorCritic(DEFAULT_LAYOUT, frame_stack=2, recurrent_hidden=32)

    def test_sequence_reevaluation_matches_rollout(self):
        buf = _rollout(self.pol)
        seq = buf.to_sequence_tensors(np.zeros(3, np.float32), 0.99, 0.95)
        L = 16
        for s in range(0, 64, L):
            for e in range(3):
                with torch.no_grad():
                    lp, _ent, v = self.pol.evaluate_sequences(
                        seq["obs"][s:s + L, e:e + 1], seq["actions"][s:s + L, e:e + 1], seq["raw_cont"][s:s + L, e:e + 1],
                        seq["hiddens"][s, e:e + 1], seq["starts"][s:s + L, e:e + 1])
                self.assertLess((lp - seq["log_probs"][s:s + L, e]).abs().max().item(), 1e-4)
                self.assertLess((v - seq["values"][s:s + L, e]).abs().max().item(), 1e-4)

    def test_memory_resets_at_episode_start(self):
        D = self.pol.frame_floats * self.pol.frame_stack
        obs = torch.randn(1, D)
        h_dirty = torch.randn(1, 32)
        with torch.no_grad():
            _, h1 = self.pol._core(self.pol._features(obs), h_dirty, torch.ones(1))
            _, h2 = self.pol._core(self.pol._features(obs), torch.zeros(1, 32), torch.zeros(1))
        self.assertTrue(torch.allclose(h1, h2))

    def test_update_runs_and_first_ratio_is_one(self):
        buf = _rollout(self.pol, T=64, E=4)
        seq = buf.to_sequence_tensors(np.zeros(4, np.float32), 0.99, 0.95)
        args = types.SimpleNamespace(n_epochs=2, minibatch_size=64, clip_coef=0.2, value_clip_coef=0.2, value_coef=0.5,
                                     entropy_coef=0.003, max_grad_norm=0.5, target_kl=0.02, learning_rate=3e-4,
                                     recurrent_seq_len=16, kl_mode="adaptive", kl_hard_factor=4.0)
        opt = torch.optim.Adam(self.pol.parameters(), lr=3e-4, eps=1e-5)
        before = {k: v.clone() for k, v in self.pol.state_dict().items()}
        st = train.ppo_update_recurrent(self.pol, opt, seq, args)
        self.assertLess(st["mb0_max_abs_logratio"], 1e-4)
        self.assertGreater(st["minibatches_used"], 0)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in self.pol.state_dict().items() if k.startswith("core.")))

    def test_checkpoint_shape_inference(self):
        self.assertEqual(recurrent_hidden_of(self.pol.state_dict()), 32)
        ff = ActorCritic(DEFAULT_LAYOUT, frame_stack=2)
        self.assertEqual(recurrent_hidden_of(ff.state_dict()), 0)


if __name__ == "__main__":
    unittest.main()
