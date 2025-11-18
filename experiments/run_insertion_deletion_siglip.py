import os
import subprocess

PATH_INPUT = "../results/mscoco"
PATH_METADATA = "../results"
MODEL_NAME = "google/siglip2-base-patch32-256"
ps = [
    0.3, 
    0.4, 
    0.5, 
    0.6, 
    0.7
]
budget = 2**14

for mode in [
    'banzhaf', 
    'shapley'
]:
    if mode == "banzhaf":
        for p_sampler in ps:
            subprocess.run([
                "sbatch", 
                "run_insertion_deletion_siglip.sh", 
                MODEL_NAME,
                os.path.join(PATH_INPUT, MODEL_NAME, str(budget), mode, str(p_sampler)),
                os.path.join(PATH_INPUT, MODEL_NAME, str(budget), mode, str(p_sampler)),
                os.path.join(PATH_METADATA, MODEL_NAME),
                mode,
                str(p_sampler),
                str(budget)
            ])
    else:
        subprocess.run([
            "sbatch", 
            "run_insertion_deletion_siglip.sh", 
            MODEL_NAME,
            os.path.join(PATH_INPUT, MODEL_NAME, str(budget), mode),
            os.path.join(PATH_INPUT, MODEL_NAME, str(budget), mode),
            os.path.join(PATH_METADATA, MODEL_NAME),
            mode,
            str(0.5),
            str(budget)
        ])