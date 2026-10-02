import sys, os, numpy as np
W=os.environ.get("OGRL_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, W+"/Tools/rl"); sys.path.insert(0, W+"/Tools/rl/ppo")
os.environ["OGRL_BINARY"]=W+"/BuildArm64/Overgrowth.app/Contents/MacOS/Overgrowth"
from env import OvergrowthEnv
from obs_schema import DEFAULT_LAYOUT as L
def run(flags, tag):
    env=OvergrowthEnv(repo_root=W, level="arenas/t_train_101.xml", shm_name=f"/ogrl_st{os.getpid()%1000}{tag}", seed=5, act_period=4, frame_stack=1,
                      extra_config_lines=["rl_target_select: 2","rl_button_edges: 1"]+flags)
    try:
        obs=env.reset(seed=5, soft=False, difficulty=0.0, opponents=1)
        f0=obs[-L.total_floats:][L.FORWARD].copy()
        vz=[];dots=[]
        for t in range(45):
            a=np.array([0,-1, 0,0,0,0,0,1],dtype=np.float32)  # stick back, walk held
            obs,r,d,i=env.step(a); f=obs[-L.total_floats:]
            vz.append(f[L.VEL_BODY][2]); dots.append(float(np.dot(f[L.FORWARD],f0)))
        print(tag, "body-forward vel mean(last20)=%.2f"%np.mean(vz[-20:]), "facing dot initial (last)=%.2f"%dots[-1])
    finally: env.close()
run([], "default")
run(["rl_stance_walk: 1"], "stance")
