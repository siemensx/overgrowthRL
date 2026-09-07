# Evaluation contract

How to measure whether this policy got better. Written 2026-09-07 after a run
where the wrong instrument said "plateau" while the right one said +13.8pp.

Follow this exactly. Every rule below exists because breaking it produced a
wrong conclusion that someone then acted on.

---

## 0. The one-line version

```bash
bash Tools/rl/comprehensive_eval.sh <tag>
```

Runs on the idle Windows trainer, takes ~45 min, writes to
`Tools/rl/runs/run21_mac/eval/comprehensive/<tag>_step<N>/`. Compare two of
those directories. Nothing else counts as evidence of improvement.

---

## 1. Never judge progress from the training win rate

`metrics.jsonl`'s win rate is measured with the **stochastic** policy against a
**curriculum-sampled** scenario mix. Both axes (`d_max`, `opponents_max`) are at
their cap, so the sampler keeps drawing the full difficulty range regardless of
competence. A flat training win rate is therefore consistent with large real
gains.

Observed 2026-09-07: training win rate sat at ~82% all night (78.7 -> 82.7 ->
81.4) while the paired eval moved +13.8pp at 1 opponent and +13.4pp at 3. The
"plateau" read from the training curve was wrong.

Training metrics are for **health** (entropy, KL, explained variance, timeout
rate, sps), never for **capability**.

---

## 2. The comprehensive eval

```bash
bash Tools/rl/comprehensive_eval.sh morning          # tag is free-form
bash Tools/rl/comprehensive_eval.sh after-wariness   # e.g. after a change
```

What it does:

1. Reads the live checkpoint, records its `global_step`.
2. Copies it to `Tools/rl/ppo/checkpoints/eval_snapshots/run21_mac_step<N>.pt`
   so the exact weights behind a number are recoverable later.
3. scp's it to `trainer-lan` and runs `multi_opponent_eval.py` there, so it
   never competes with training on the Mac.
4. Pulls per-cell JSONs back into
   `Tools/rl/runs/run21_mac/eval/comprehensive/<tag>_step<N>/`.

Fixed parameters -- **do not change these between runs you intend to compare**:

| parameter | value | why |
|---|---|---|
| `--levels` | `arenas/t_train_101.xml arenas/t_held_203.xml` | one SEEN, one HELD OUT. Both verified: original geometry, RL level script, spawn height gap 0.0u |
| `--opponents` | `1 2 3` | the three tiers |
| `--difficulty-bands` | `0.2,0.5,0.8,1.0` | 40 episodes **per band**, so 160/cell, 960 total |
| `--episodes` | `40` | |
| `--max-episode-steps` | `1200` | MUST match training's cap. evaluate.py defaults to 900; a timeout counts as a loss, so a shorter cap silently penalises exactly the long multi-opponent fights |
| `--seed-base` | `8900000` | fixed, so reruns are **paired** -- same scenarios, and the diff is the policy |

Deterministic (greedy) actions by default. Greedy is worth roughly +4-6 points
over stochastic; that is fine as long as both sides of a comparison are greedy.

---

## 3. Reading the result

```bash
python3 - <<'PY'
import json, glob, os, math
A="Tools/rl/runs/run21_mac/eval/comprehensive/<older>"
B="Tools/rl/runs/run21_mac/eval/comprehensive/<newer>"
def load(d):
    c={}
    for f in glob.glob(os.path.join(d,"t_*.json")):
        j=json.load(open(f)); lv,o=os.path.basename(f)[:-5].rsplit("_",1)
        c[(lv,int(o))]=j
    return c
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); m=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-m)/d,(c+m)/d)
a,b=load(A),load(B)
for o in (1,2,3):
    ka=na=kb=nb=0
    for lv in sorted({k[0] for k in a}):
        for bd in a[(lv,o)]['bands']:
            x=bd['policy']['outcomes']; ka+=x['won']; na+=sum(x.values())
        for bd in b[(lv,o)]['bands']:
            x=bd['policy']['outcomes']; kb+=x['won']; nb+=sum(x.values())
    ca,cb=wilson(ka,na),wilson(kb,nb)
    print(f"{o} opp: {ka/na:.3f} [{ca[0]:.2f},{ca[1]:.2f}] -> {kb/nb:.3f} [{cb[0]:.2f},{cb[1]:.2f}]  {kb/nb-ka/na:+.3f}")
PY
```

**Report per band and per opponent count, never a single pooled number.** A
pooled figure over a difficulty curriculum hides where the change happened; the
2026-09-07 gain was concentrated in the hardest cells (1v1 at difficulty 1.0
went 0.675 -> 0.950 while difficulty 0.2 was already saturated).

n=480 per opponent count across both maps. Treat a change as real only when the
**95% Wilson intervals do not overlap**. At n=480 that needs roughly 6pp.

Two points are not a trend. Three or more before claiming a direction --
2026-09-06 produced a "regression", then an escalation with bigger n that only
tightened the error bars on a single point, then a withdrawal when the third
point showed recovery.

---

## 4. The stopping rule

Run the eval every ~50M steps. **Stop training and change the environment when
two consecutive evals move less than 3pp** at 2 and 3 opponents. That is the
saturation test. A fixed step target is not.

Reference standings at 247.3M (2026-09-07), all-map aggregate:

| | 1 opp | 2 opp | 3 opp |
|---|---|---|---|
| 177.7M | 0.787 | 0.691 | 0.428 |
| 247.3M | **0.925** | **0.741** | **0.562** |

---

## 5. Behavioural checks the win rate cannot see

A win rate says nothing about *how* it wins. Run these alongside.

**Knockouts scored per episode** -- distinguishes "ground down by the third
opponent" from "knocked out by the first", which look identical in a win rate:

```bash
python3 - <<'PY'
import json
from collections import defaultdict
rows=[json.loads(l) for l in open('Tools/rl/runs/run21_mac/episodes.jsonl',errors='replace') if l.strip()]
recent=rows[-15000:]
kos=lambda r:(r.get('components') or {}).get('hostile_kos_this_step',0.0)
for o in (1,2,3):
    s=[r for r in recent if r.get('opponents')==o]
    d=defaultdict(int)
    for r in s: d[int(round(kos(r)))]+=1
    n=len(s); w=sum(1 for r in s if r.get('outcome')=='won')
    print(f"{o} opp n={n} win={w/n:.3f} meanKO={sum(map(kos,s))/n:.2f} "
          + " ".join(f"{k}KO={100*d[k]/n:.1f}%" for k in range(o+1)))
PY
```

At 247M: 1v3 losses were 40.2% with zero knockouts, 41.0% with one, 18.8% with
two. Two-fifths of losses are shutouts, not close fights.

**Move distribution** -- the ground truth on repertoire collapse. Requires
`rl_log_attacks` in the config; parses the engine's `RLATK` lines:

```bash
python3 Tools/rl/move_stats.py
```

At 247M the policy is ~90% legcannon. Attack *attempts* and attack *outcomes*
differ sharply -- report the one you mean.

**Entropy** from `metrics.jsonl` (`ppo.entropy`, random reference ~7.0) is the
cheapest repertoire-collapse signal: 2.36 -> 1.24 over one night is the jump-kick
commitment, quantified.

---

## 6. Map validity -- check before trusting any number

A map where the fighters cannot reach each other produces episodes that run to
the cap and count as losses. On 2026-09-06 this made 29.2% of all training steps
worthless and made every quoted win rate an average over functional and
non-functional levels.

```bash
python3 Tools/rl/validate_maps.py --checkpoint <ckpt> --levels arenas/<map>.xml --episodes 14
```

Gate: **timeout rate <= 3%** at every opponent count. Healthy maps sit at
0.2-2.5%.

Headless is not sufficient. A level can train fine for a day and kill the window
when a human opens it, because training never calls `Engine::DrawScene`:

```bash
bash Tools/rl/render_smoke.sh <ckpt> arenas/<map>.xml
```

A crashing engine makes `watch.py` hang silently rather than print an error --
**treat "hung with no output" as a crash**. macOS deduplicates crash reports, so
a detector keyed on a new `.ips` file only fires the first time a signature
appears.

Also verify the level script. `<Script>` must be
`Data/Scripts/arena_level_1v1_unarmed.as`. The stock `arena_level.as` revives
the fallen and runs multiple rounds, which makes a non-timeout meaningless:

```bash
grep -o '<Script>[^<]*</Script>' "$OG_DATA/Levels/arenas/<map>.xml"
```

---

## 7. Watching it fight

```bash
python3 Tools/rl/ppo/watch.py --checkpoint Tools/rl/ppo/checkpoints/run21_mac.pt \
  --level arenas/t_train_101.xml --opponents 3 --difficulty 0.6 \
  --frame-stack 4 --act-period 4 --episodes 5 --max-episode-real-seconds 100
```

Caveats:
* **Five episodes cannot distinguish a broken policy from a normal one.** At a
  0.4 per-episode win rate, P(0 wins in 5) = 7.5% and P(4 wins in 5) = 8.0%.
  Both tails were observed on 2026-09-06 and one was nearly acted on.
* Running this while training runs costs a training restart under memory
  pressure. Expect it.

---

## 8. Machine hygiene

* Verify a training restart by the **shm prefix** (`ps aux | grep train_vec`),
  never by an exit code. Two restarts on 2026-09-07 silently did not happen.
* Never `pkill -f "MacOS/Overgrowth"` -- it matches the training workers. Probe
  scripts match `write-dir.*env-ogrl_w` instead.
* Regenerating a map invalidates the baked navmeshes cached in
  `.rl_write_dirs/`. Wipe that directory on any geometry change.
* Only one eval may drive the trainer at a time. `multi_opponent_eval.py` writes
  per-invocation subdirectories now, but two concurrent runs still split the
  cores and halve both.
