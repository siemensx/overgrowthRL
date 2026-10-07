# Training profiles (OGRL-20261007-008)

One JSON per training run. `run_profile.py` turns a profile into a `train_vec.py` command and supervises
it; `remote/switch_profile.sh` switches the Windows trainer from one profile to another.

## Switch the trainer (from the Mac)

```bash
Tools/rl/remote/switch_profile.sh --status                       # what is training now
Tools/rl/remote/switch_profile.sh run39_personas_blind --dry-run # every step, nothing changed
Tools/rl/remote/switch_profile.sh run39_personas_blind           # do it
Tools/rl/remote/switch_profile.sh run38_attn_last                # go back: resumes run38 where it stopped
Tools/rl/remote/switch_profile.sh --stop                         # stop gracefully, start nothing
HOST=trainer-ts Tools/rl/remote/switch_profile.sh ...            # over Tailscale
```

Commit and push first: the trainer pulls the branch, it never receives files any other way (except maps,
which live in the purchased Steam tree and are copied with an MD5 check).

## Profiles

| profile | starts from | what changes vs run38 | the question |
|---|---|---|---|
| `run38_attn_last` | its own checkpoint | nothing (the run training now) | how high does this version climb? |
| `run39_personas_blind` | fork of run38 | 50% of fights vs patient / passive / berserker / expert / mixed opponents; 30% with the AI's intent fields hidden | does it learn to start fights against opponents that wait (it times out against them and against a standing human)? |
| `run40_horde_armed` | fork of run38 | 12 horde maps, opponents unlock up to 1v7, armed ladder from B1 | how outnumbered and how armed can the enemies get before it stops winning? |
| `run41_everything` | fork of run38 | run39 + run40 together | only after each axis has been shown learnable on its own |
| `run42_hard_fights` | fork of run38 | only `--d-min 0.7`: every fight at difficulty 0.7–1.0 (run38 spends 7.8% of fights in the benchmark's cell) | does training where the benchmark is break the 1v3 plateau? |

A fork freezes a copy of the parent's checkpoint as `<run_id>_seed.pt` at its first launch (after the
parent has been stopped), so the parent can later be resumed unchanged.

## Judging a run

* `canonical_eval.py --controls v6-omni` (suite v2): did the old skill hold?
* `robust_eval.py --controls v6-omni` (suite r1): personas, hidden intent, armed, 1v4..1v7.
* The in-training `personas` block in `metrics.jsonl` (win rate per persona and with/without intent) is
  a health signal from the sampled policy, not a result.

## Format

```json
{
  "base": "_base_v6_attn.json",     // shared learner/engine args; profile args are appended (last wins)
  "run_id": "run39_personas_blind", // checkpoint = ppo/checkpoints/<run_id>.pt, telemetry = runs/<run_id>/
  "seed": 39, "shm_tag": "r39",     // RNG seed; shared-memory prefix tag (fresh suffix every launch)
  "seed_from": "run38_attn_last.pt",// null = fresh network with the base's fresh_args
  "requires_scenario": "rl_persona",// a string the trainer's level script must contain
  "levels": "arenas/t_horde_{301..312}.xml",  // optional, overrides the base's 24 t_train maps
  "args": ["--persona-mix", "..."]
}
```

`run_profile.py --check` refuses a profile whose engine count is not a multiple of its map count
(AGENTS.md invariant 6), whose maps lack the 1vN spawn group its `--opponents-cap` needs (the fight
would silently fall back to 1v3 and become unwinnable), or whose level script is too old.
