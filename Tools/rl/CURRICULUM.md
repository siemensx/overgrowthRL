# The curriculum

Three independent gates, each measuring a different thing, each advancing on
its own evidence. None can promote another. Positions are checkpointed; the
outcome windows behind them are not, so a resume re-earns its next step.

| axis | state | advances when |
|---|---|---|
| difficulty | `d_max` 0.15 → 1.00 | solo win rate ≥ `gate_win_rate` over the top band |
| opponent count | `opponents_max` 1 → 3 | win rate at the current max ≥ `opp_gate_win_rate` |
| **armed opponents** | `armed_stage` 0 → 6 | **armed-round** win rate ≥ `armed_gate_win_rate` |

Difficulty and opponent count are both **at their cap** (1.00 and 3). The armed
ladder is the live axis.

---

## The armed ladder

`Tools/rl/curriculum.py : ARMED_STAGES`

| # | stage | armed hostiles | weapon | throw aggr. | species | min opponents |
|---|---|---|---|---|---|---|
| 0 | A unarmed | 0 | – | 1.0 (stock) | guard/raider | 1 |
| 1 | B1 1-armed-of-2 | 1 | knife | 4.0 | guard/raider | 2 |
| 2 | B2 2-armed-of-2 | 2 | knife | 4.0 | guard/raider | 2 |
| 3 | B3 1-armed-of-3 | 1 | knife | 4.0 | guard/raider | 3 |
| 4 | B4 2-armed-of-3 | 2 | random | 4.0 | guard/raider | 3 |
| 5 | B5 3-armed-of-3 | 3 | random | 4.0 | guard/raider | 3 |
| 6 | C cats + mixed | 3 | random | 6.0 | guard/raider/**cat** | 3 |

weapon: 0 random · 1 knife · 2 big_sword · 3 sword · 4 spear
species: 0 guard/raider · 5 cat · 6 random incl. cat

### What each dial actually does

**`armed_count`** — how many of the hostiles carry a weapon. The agent is
**never** armed by this axis; the whole point is an unarmed agent against a
partly-armed group. The stock level script armed *every* character including
the agent, which was fixed in `gen_1v1_scenario.py` (anchor 6b).

**`throw_aggression > 1.0`** turns on the Dynamic AI Aggression behaviours
ported from workshop mod 1241215386:

* armed bots **stop rolling away** from a jump kick. The roll is not a counter
  — measured, it costs the policy 15 points at difficulty 0.6 and *nothing* at
  1.0, where the block-skill term is already saturated — and it actively
  defeats the throw, by disengaging to range 3-4 while the throw predicate
  needs close range. Armed bots hold at 1.5-3.0 and throw instead.
* the throw fires on `sub_goal == _avoid_jump_kick`, which is set **precisely
  when the target is airborne**. A knife thrown at a committed jump kick is the
  first thing in this game that punishes the policy's only strategy.
* rabbits throw a knife when hurt and at range; bots throw while airborne.

**`species` 5/6** brings in cats, whose controller throws on its own account
and prioritises airborne targets even without the DAA rules.

### Anti-forgetting

A stage with `min opponents` > 1 does **not** arm the solo rounds. `opp_keep_solo`
keeps 35% of episodes at 1v1 and those stay unarmed, so the 1v1 competence the
eval tracks is still measured against the same opponent it always was.

### How advancement is recorded

Only **armed** rounds feed the gate — an unarmed solo round cannot promote a
ladder whose fights it never had. On advance the window is cleared, so the next
stage earns its own `armed_gate_min_samples` (150) armed episodes.

Every transition is written three ways:

* a `curriculum_advance` event in `runs/run21_mac/events.jsonl`, **stamped with
  the global step**, e.g. `armed stage 1 -> 2 (B2 2-armed-of-2) at
  global_step=261,493,468 d_max=1.00 opponents_max=3`
* a `[curriculum] global_step=... ARMED STAGE 1 -> 2 B2 2-armed-of-2` line in
  the run log
* `armed_stage` and `armed_stage_label` on every metrics row, and
  `armed_count` / `weapon_type` on every episode row

```bash
# every stage transition this run has made, with its step
grep curriculum_advance Tools/rl/runs/run21_mac/events.jsonl
```

---

## Watching any stage on the current checkpoint

`--stage N` sets all four dials from the table and raises `--opponents` to that
stage's minimum.

```bash
python3 Tools/rl/ppo/watch.py --checkpoint Tools/rl/ppo/checkpoints/run21_mac.pt --level arenas/t_train_101.xml --stage 0 --difficulty 0.8 --frame-stack 4 --act-period 4 --episodes 5 --max-episode-real-seconds 100
```

Swap `--stage` for any row: `1` 1-armed-of-2, `2` 2-armed-of-2, `3`
1-armed-of-3, `4` 2-armed-of-3, `5` 3-armed-of-3, `6` cats + mixed weapons.

Hand-set a scenario instead of a stage — e.g. three spear-carriers:

```bash
python3 Tools/rl/ppo/watch.py --checkpoint Tools/rl/ppo/checkpoints/run21_mac.pt --level arenas/t_train_101.xml --opponents 3 --armed-count 3 --weapon-type 4 --throw-aggression 6.0 --difficulty 1.0 --frame-stack 4 --act-period 4 --episodes 5 --max-episode-real-seconds 100
```

Count the throws rather than watching them:

```bash
python3 Tools/rl/probe_throws.py --checkpoint Tools/rl/ppo/checkpoints/run21_mac.pt --episodes 3 --opponents 3 --difficulty 0.8 --armed-count 3 --weapon-type 1 --throw-aggression 4.0 --max-episode-steps 400 --shm-name /ogrl_look
```

It reports throws committed, how many were at an **airborne** agent, throw
distance, and the agent's airborne time per jump.

> Running any of these while training runs costs a worker under memory
> pressure. Expect it.

---

## Measured, so it is not taken on faith

* `GetThrowTarget()` aimed by the **camera's** facing, so headless AI returned
  no target on **1220/1220** calls. No AI had ever thrown a weapon in this
  project. Fixed to aim from the thrower's own facing: `0/1220 -> 25/489`
  valid, **4 throws, 4 at an airborne agent (100%)**, at 7.9 units.
* Agent airborne time: **36-39 steps, 1.2-1.3 s per jump**, 6-9 jumps per
  episode — ample window to be hit.
* Entropy was pinned at its 0.003 floor since step 10M. Raised to 0.012
  decaying to 0.008; entropy 1.24 -> 1.80 within an hour.

## Dead ends, so they are not retried

* **Jump-kick wariness is not an axis.** Seeding
  `got_hit_by_leg_cannon_count` looked like a lever (win 1.000 -> 0.850 at
  difficulty 0.6) but at difficulty 1.0 wariness 0 and 8 are identical —
  `cureved_game_difficulty` is 1.0 there, so the block-skill term is already
  saturated. Past saturation it stops making bots better and makes them refuse
  to fight: 0 wins and 19/20 timeouts at 3 opponents.
* **Wolves are the wrong "harder species".** The legcannon has a hard-coded
  `block_health = 0.0f` bypass and an outright kill on a wolf at `ko_shield == 1`.
  It is the *designed* anti-wolf tool; wolves would reinforce the monoculture.
