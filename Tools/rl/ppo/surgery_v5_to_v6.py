#!/usr/bin/env python3
"""Transplant a schema-v5 checkpoint into the schema-v6 network (OGRL-20261002-019).

v6 appends fields at the end of the self block (+11) and of each entity slot (+21), so every v5 input
has a v6 home. Two v5 inputs changed meaning in v6 and are handled exactly:
  * "right" was mirrored in v5 (rl_observation.cpp). In v6 the lateral components flip sign:
    entity rel_pos.x (field 2), rel_vel.x (5), fwd.x (24) -> negate their first-layer weight columns
    and their normaliser means. The 16 geometry rays are mirrored: v6 ray k == v5 ray (16-k) % 16
    -> permute weight columns and normaliser stats.
  * `grounded` was read as 4 bytes of a 1-byte bool in v5; v6 reads the bool. Same meaning, kept.
New inputs get zero weights (mean 0, var 1 in the normaliser), so the transplanted policy computes
exactly what the v5 policy computed on the equivalent v5 observation -- verified by --check.
The optimizer state is reset (first-layer shapes changed); everything else carries over.

    python3 Tools/rl/ppo/surgery_v5_to_v6.py --src run27_win.pt --dst run27_v6.pt --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from obs_schema import DEFAULT_LAYOUT as V6, ENTITY_FLOATS as E6, SELF_V6_FLOATS, ENTITY_V6_FLOATS  # noqa: E402
from policy import ActorCritic  # noqa: E402
from normalize import ObservationNormalizer  # noqa: E402

P5, E5, H, R, N = 35, 33, 24, 16, 8          # v5 self, entity, history, rays, entity slots
P6 = P5 + SELF_V6_FLOATS
assert E6 == E5 + ENTITY_V6_FLOATS
ENT_FLIP = [2, 5, 24]                         # lateral components whose sign flipped in v6
RAY_SRC = [(-k) % R for k in range(R)]        # v6 ray k takes v5 ray (16-k)%16


class _V5Layout:
    """Minimal v5 layout so ActorCritic / ObservationNormalizer can be built for the source."""
    max_visible_entities = N
    entities_start = P5 + H
    total_floats = P5 + H + N * E5 + R

    def entity_slice(self, slot):
        s = self.entities_start + slot * E5
        return slice(s, s + E5)


def _non_entity_map(fs: int) -> tuple[list[int], list[float]]:
    """For each v6 non-entity input column (frame-major), the v5 column it copies (-1 = new)."""
    ne5, ne6 = P5 + H + R, P6 + H + R
    src = []
    for f in range(fs):
        for j in range(ne6):
            if j < P5:
                i = j
            elif j < P6:
                i = -1
            elif j < P6 + H:
                i = j - P6 + P5
            else:
                i = P5 + H + RAY_SRC[j - P6 - H]
            src.append(-1 if i < 0 else f * ne5 + i)
    return src


def transplant(ck: dict) -> dict:
    fs = int(ck["frame_stack"])
    v5 = _V5Layout()
    old = ActorCritic(v5, frame_stack=fs)
    old.load_state_dict(ck["policy"])
    new = ActorCritic(V6, frame_stack=fs)
    sd_old, sd_new = old.state_dict(), new.state_dict()
    out = {}
    for k, v in sd_new.items():
        if k not in sd_old:
            out[k] = v  # (none expected for the feed-forward policy)
            continue
        if sd_old[k].shape == v.shape:
            out[k] = sd_old[k].clone()
    # entity encoder first layer: (64, 33) -> (64, 54)
    w = sd_old["entity_encoder.mlp.0.weight"]
    w6 = torch.zeros(w.shape[0], E6)
    w6[:, :E5] = w
    w6[:, ENT_FLIP] = -w6[:, ENT_FLIP]
    # v5 fed the normalised `valid` flag (~1.8e-7, the rounding accident); v6 feeds the raw 1/0. The
    # flag's contribution was therefore ~0 in v5 -- keep it ~0 by zeroing its column.
    w6[:, 0] = 0.0
    out["entity_encoder.mlp.0.weight"] = w6
    # proprioception branch first layer: (256, 75*fs) -> (256, 86*fs)
    w = sd_old["proprioception_branch.0.weight"]
    src = _non_entity_map(fs)
    w6 = torch.zeros(w.shape[0], len(src))
    for j, i in enumerate(src):
        if i >= 0:
            w6[:, j] = w[:, i]
    out["proprioception_branch.0.weight"] = w6
    missing = [k for k in sd_new if k not in out]
    assert not missing, missing
    new.load_state_dict(out)

    # normaliser
    on = ck["obs_normalizer"]
    em, ev = np.asarray(on["entity_mean"], np.float64), np.asarray(on["entity_var"], np.float64)
    em6, ev6 = np.zeros(E6), np.ones(E6)
    em6[:E5], ev6[:E5] = em, ev
    em6[ENT_FLIP] = -em6[ENT_FLIP]
    nm, nv = np.asarray(on["non_entity_mean"], np.float64), np.asarray(on["non_entity_var"], np.float64)
    nm6, nv6 = np.zeros(len(src)), np.ones(len(src))
    for j, i in enumerate(src):
        if i >= 0:
            nm6[j], nv6[j] = nm[i], nv[i]
    norm6 = {"entity_mean": em6, "entity_var": ev6, "entity_count": on["entity_count"],
             "non_entity_mean": nm6, "non_entity_var": nv6, "non_entity_count": on["non_entity_count"]}

    opt = torch.optim.Adam(new.parameters(), lr=3e-4, eps=1e-5)
    res = dict(ck)
    res.update({"policy": new.state_dict(), "obs_normalizer": norm6, "optimizer": opt.state_dict(),
                "layout_total_floats": V6.total_floats,
                "surgery": {"from_schema": 5, "to_schema": 6, "experiment": "OGRL-20261002-019",
                            "source_global_step": int(ck.get("global_step", -1))}})
    return res, old, new


def v5_to_v6_obs(x5: np.ndarray, fs: int) -> np.ndarray:
    """Map a raw stacked v5 observation to the raw v6 observation of the same world state
    (new fields zero, lateral signs flipped, rays mirrored)."""
    v5 = _V5Layout()
    out = np.zeros((x5.shape[0], fs * V6.total_floats), np.float32)
    for f in range(fs):
        a = x5[:, f * v5.total_floats:(f + 1) * v5.total_floats]
        b = out[:, f * V6.total_floats:(f + 1) * V6.total_floats]
        b[:, :P5] = a[:, :P5]
        b[:, P6:P6 + H] = a[:, P5:P5 + H]
        for s in range(N):
            e5 = a[:, v5.entities_start + s * E5: v5.entities_start + (s + 1) * E5]
            o = V6.entities_start + s * E6
            b[:, o:o + E5] = e5
            b[:, [o + k for k in ENT_FLIP]] *= -1
        r5 = a[:, v5.entities_start + N * E5:]
        b[:, V6.rays_start:V6.rays_start + R] = r5[:, RAY_SRC]
    return out


def check(ck: dict, res: dict, old, new, n: int = 256) -> float:
    fs = int(ck["frame_stack"])
    v5 = _V5Layout()
    rng = np.random.default_rng(0)
    x5 = rng.normal(size=(n, fs * v5.total_floats)).astype(np.float32)
    for f in range(fs):   # valid flags: first 2 slots real, rest padding (zero)
        base = f * v5.total_floats + v5.entities_start
        for s in range(N):
            x5[:, base + s * E5: base + (s + 1) * E5] *= (1.0 if s < 2 else 0.0)
            x5[:, base + s * E5] = 1.0 if s < 2 else 0.0
    n5 = ObservationNormalizer(v5, frame_stack=fs); n5.load_state_dict(ck["obs_normalizer"])
    n6 = ObservationNormalizer(V6, frame_stack=fs); n6.load_state_dict(res["obs_normalizer"])
    o5 = n5.normalize(x5, update=False)
    # This branch's normaliser writes the raw valid flag (1/0) back; the v5 policy was trained on the
    # NORMALISED flag ((1 - mean)/std ~ 1.8e-7 for run21/27). Reproduce what it actually saw.
    on = ck["obs_normalizer"]
    vnorm = (1.0 - float(on["entity_mean"][0])) / float(np.sqrt(on["entity_var"][0] + 1e-8))
    fr = o5.reshape(n, fs, v5.total_floats)
    for s_ in range(N):
        col = v5.entities_start + s_ * E5
        fr[:, :, col] = np.where(fr[:, :, col] > 0.5, vnorm, 0.0)
    o5 = torch.as_tensor(fr.reshape(n, -1), dtype=torch.float32)
    o6 = torch.as_tensor(n6.normalize(v5_to_v6_obs(x5, fs), update=False), dtype=torch.float32)
    with torch.no_grad():
        m5, ls5, l5 = old.actor_params(o5)
        m6, ls6, l6 = new.actor_params(o6)
        v5v, v6v = old.get_value(o5), new.get_value(o6)
    err = max((m5 - m6).abs().max().item(), (l5 - l6).abs().max().item(), (v5v - v6v).abs().max().item())
    return err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    assert ck.get("layout_total_floats") == _V5Layout.total_floats, "source must be schema v5 (339 floats)"
    res, old, new = transplant(ck)
    if a.check:
        err = check(ck, res, old, new)
        print(f"max |v5 - v6| over actor means, button logits and value: {err:.3e}")
        if err > 1e-4:
            print("MISMATCH -- not writing")
            return 1
    if Path(a.dst).exists():
        print(f"refusing to overwrite {a.dst}")
        return 2
    torch.save(res, a.dst)
    print(f"wrote {a.dst} (global_step {res.get('global_step')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
