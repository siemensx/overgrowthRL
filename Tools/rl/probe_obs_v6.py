"""Smoke test obs schema v6 on the v6 worktree engine: 1v3, random-ish actions, dump field stats."""
import sys, os
import numpy as np
W=os.environ.get("OGRL_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, W+"/Tools/rl"); sys.path.insert(0, W+"/Tools/rl/ppo")
os.environ["OGRL_BINARY"]=W+"/BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth"
from env import OvergrowthEnv
from obs_schema import DEFAULT_LAYOUT as L, ENTITY_FLOATS
omni = len(sys.argv)>1 and sys.argv[1]=="omni"
flags=["rl_target_select: 2","rl_button_edges: 1","rl_no_feint: 1","rl_stick_deadzone: 0.3","rl_stance_walk: 1"]+(["rl_obs_omniscient: 1"] if omni else [])
env=OvergrowthEnv(repo_root=W, level="arenas/t_train_101.xml", shm_name=f"/ogrl_v6s{os.getpid()%10000}", seed=11, act_period=4, frame_stack=1, extra_config_lines=flags)
rng=np.random.default_rng(0)
stats={k:[] for k in ["n_valid","targets_me","ai_attacking","group_wait","will_counter","goal_attack","subgoal_sum","los","hostiles_awake","grounded","feinting","vel_body_mag","vel_world_mag"]}
try:
    obs=env.reset(seed=11, soft=False, difficulty=1.0, opponents=3)
    for t in range(600):
        a=np.concatenate([rng.uniform(-1,1,2), (rng.random(6)>0.7).astype(np.float32)]).astype(np.float32)
        obs,r,done,info=env.step(a)
        f=obs[-L.total_floats:]
        stats["hostiles_awake"].append(f[L.HOSTILES_AWAKE]); stats["grounded"].append(f[L.GROUNDED]); stats["feinting"].append(f[L.FEINTING])
        stats["vel_body_mag"].append(np.linalg.norm(f[L.VEL_BODY])); stats["vel_world_mag"].append(np.linalg.norm(f[L.VEL]))
        nv=0
        for s in range(L.max_visible_entities):
            e=f[L.entity_slice(s)]
            if e[0]<0.5: continue
            nv+=1
            stats["targets_me"].append(e[L.E_TARGETS_ME]); stats["ai_attacking"].append(e[L.E_AI_ATTACKING]); stats["group_wait"].append(e[L.E_GROUP_WAIT])
            stats["will_counter"].append(e[L.E_WILL_THROW_COUNTER]); stats["goal_attack"].append(e[L.E_GOAL_ATTACK]); stats["subgoal_sum"].append(e[L.E_SUB_GOAL].sum()); stats["los"].append(e[L.E_LINE_OF_SIGHT])
        stats["n_valid"].append(nv)
        if done:
            obs=env.reset(seed=12+t, soft=False, difficulty=1.0, opponents=3)
finally:
    env.close()
print("omniscient" if omni else "LOS", "total_floats", L.total_floats)
for k,v in stats.items():
    v=np.asarray(v,dtype=float); print(f"{k:16} n={len(v):5} mean={v.mean() if len(v) else float('nan'):.3f} min={v.min() if len(v) else 0:.2f} max={v.max() if len(v) else 0:.2f}")
assert abs(np.mean(stats["vel_body_mag"])-np.mean(stats["vel_world_mag"]))<0.05, "body-frame velocity must preserve magnitude (horizontal-only rotation)"
