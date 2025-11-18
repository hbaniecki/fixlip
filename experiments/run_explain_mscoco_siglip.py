import os
import subprocess

PATH_INPUT = "../results"
PATH_OUTPUT = "../results/mscoco"
MODEL_NAME = "google/siglip2-base-patch32-256"
RANDOM_STATE = 0
START = 900
STOP = 1000
budgets = [
    2**15, 
    2**16, 
    2**17, 
    2**18, 
    2**19,
    2**20, 
    2**21
]
ps = [
    # 0.3, 
    # 0.4, 
    0.5,
    # 0.6, 
    # 0.7
]
batch_size = 64

for budget in budgets:
    for mode in [
        'fixlip', 
        'banzhaf', 
        # 'shapley'
    ]:
        if not mode.startswith("s"):
            for p_sampler in ps:
                subprocess.run([
                    "sbatch", 
                    "run_explain_mscoco_siglip.sh", 
                    MODEL_NAME,
                    os.path.join(PATH_INPUT, MODEL_NAME),
                    os.path.join(PATH_OUTPUT, MODEL_NAME, str(budget), mode, str(p_sampler)),
                    str(START),
                    str(STOP),
                    mode,
                    str(p_sampler),
                    str(budget),
                    str(batch_size),
                    str(RANDOM_STATE)
                ])
        else:
            subprocess.run([
                "sbatch", 
                "run_explain_mscoco_siglip.sh", 
                MODEL_NAME,
                os.path.join(PATH_INPUT, MODEL_NAME),
                os.path.join(PATH_OUTPUT, MODEL_NAME, str(budget), mode),
                str(START),
                str(STOP),
                mode,
                str(0.5),
                str(budget),
                str(batch_size),
                str(RANDOM_STATE)
            ])