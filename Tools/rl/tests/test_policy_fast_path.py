"""The policy's action path computes Normal/Bernoulli log-prob and entropy
directly instead of constructing torch.distributions objects -- at this batch
size 58% of the call was object plumbing rather than math (see policy.py's
module note for the measurement).

That is only safe if the arithmetic is IDENTICAL, so these tests pin the
written-out formulas against torch.distributions itself. That is exactly why
policy.py still imports Normal and Bernoulli.
"""
from __future__ import annotations

import os
import sys
import unittest

import torch
from torch.distributions import Normal, Bernoulli

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ppo"))
sys.path.insert(0, os.path.join(HERE, ".."))

from policy import (  # noqa: E402
    ActorCritic,
    _normal_log_prob,
    _normal_entropy,
    _bernoulli_log_prob,
    _bernoulli_entropy,
)
from obs_schema import DEFAULT_LAYOUT  # noqa: E402

TOL = 1e-6


def _policy():
    torch.manual_seed(7)
    return ActorCritic(DEFAULT_LAYOUT, frame_stack=4)


class TestDistributionFormulas(unittest.TestCase):
    def test_normal_log_prob_matches_torch(self):
        for shape in [(1, 2), (4, 2), (256, 2)]:
            with self.subTest(shape=shape):
                torch.manual_seed(0)
                mean = torch.randn(shape)
                log_std = torch.randn(shape).clamp(-5.0, 2.0)
                value = torch.randn(shape)
                ref = Normal(mean, log_std.exp()).log_prob(value)
                got = _normal_log_prob(value, mean, log_std)
                self.assertTrue(torch.allclose(got, ref, atol=TOL, rtol=TOL))

    def test_normal_entropy_matches_torch(self):
        torch.manual_seed(1)
        log_std = torch.randn(64, 2).clamp(-5.0, 2.0)
        ref = Normal(torch.zeros_like(log_std), log_std.exp()).entropy()
        self.assertTrue(torch.allclose(_normal_entropy(log_std), ref, atol=TOL, rtol=TOL))

    def test_bernoulli_log_prob_matches_torch(self):
        for shape in [(1, 6), (4, 6), (256, 6)]:
            with self.subTest(shape=shape):
                torch.manual_seed(2)
                logits = torch.randn(shape) * 3.0
                value = torch.bernoulli(torch.sigmoid(logits))
                ref = Bernoulli(logits=logits).log_prob(value)
                got = _bernoulli_log_prob(value, logits)
                self.assertTrue(torch.allclose(got, ref, atol=TOL, rtol=TOL))

    def test_bernoulli_entropy_matches_torch(self):
        torch.manual_seed(3)
        logits = torch.randn(128, 6) * 3.0
        ref = Bernoulli(logits=logits).entropy()
        self.assertTrue(torch.allclose(_bernoulli_entropy(logits), ref, atol=TOL, rtol=TOL))

    def test_extreme_logits_do_not_diverge(self):
        """binary_cross_entropy_with_logits is the numerically stable form. A
        naive log(sigmoid(x)) would return -inf here and poison the loss."""
        logits = torch.tensor([[-60.0, -20.0, 0.0, 20.0, 60.0, 1e3]])
        value = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0]])
        lp = _bernoulli_log_prob(value, logits)
        ent = _bernoulli_entropy(logits)
        self.assertTrue(torch.isfinite(lp).all())
        self.assertTrue(torch.isfinite(ent).all())
        ref = Bernoulli(logits=logits).log_prob(value)
        self.assertTrue(torch.allclose(lp, ref, atol=TOL, rtol=TOL))


class TestPolicyFastPath(unittest.TestCase):
    def test_update_path_matches_distribution_implementation(self):
        """The PPO update re-evaluates a GIVEN action. This path must agree with
        the torch.distributions computation to float precision, or the
        importance ratio -- and every gradient -- changes."""
        pol = _policy()
        obs = torch.randn(16, DEFAULT_LAYOUT.total_floats * 4)
        with torch.no_grad():
            act, _, _, _ = pol.get_action_and_value(obs)
            _, fast_lp, fast_ent, fast_val = pol.get_action_and_value(obs, act)

            feats = pol._features(obs)
            cdist, ddist = pol._distributions_from_features(feats)
            cont = act[..., :2].clamp(-1.0 + 1e-3, 1.0 - 1e-3)
            raw = torch.atanh(cont)
            ref_lp = (cdist.log_prob(raw) - torch.log(1.0 - cont.pow(2) + 1e-3)).sum(-1) \
                + ddist.log_prob(act[..., 2:]).sum(-1)
            ref_ent = cdist.entropy().sum(-1) + ddist.entropy().sum(-1)
            ref_val = pol.critic_out(pol.critic_trunk(feats)).squeeze(-1)

        self.assertTrue(torch.allclose(fast_lp, ref_lp, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(fast_ent, ref_ent, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(fast_val, ref_val, atol=1e-6, rtol=1e-6))

    def test_sampled_actions_are_well_formed(self):
        pol = _policy()
        obs = torch.randn(64, DEFAULT_LAYOUT.total_floats * 4)
        with torch.no_grad():
            act, lp, ent, val = pol.get_action_and_value(obs)
        self.assertEqual(tuple(act.shape), (64, 8))
        self.assertTrue((act[:, :2] >= -1.0).all() and (act[:, :2] <= 1.0).all())
        self.assertTrue(torch.isin(act[:, 2:], torch.tensor([0.0, 1.0])).all())
        for t in (lp, ent, val):
            self.assertTrue(torch.isfinite(t).all())
            self.assertEqual(tuple(t.shape), (64,))

    def test_entropy_is_per_sample_not_a_broadcast_scalar(self):
        """continuous_log_std is one learned parameter shared across the batch.
        A careless expand() there would collapse entropy to a single value and
        quietly change the entropy bonus; it must still vary with the discrete
        head, which is per-sample."""
        pol = _policy()
        obs = torch.randn(32, DEFAULT_LAYOUT.total_floats * 4)
        with torch.no_grad():
            _, _, ent, _ = pol.get_action_and_value(obs)
        self.assertEqual(tuple(ent.shape), (32,))
        self.assertGreater(float(ent.std()), 0.0)


if __name__ == "__main__":
    unittest.main()
