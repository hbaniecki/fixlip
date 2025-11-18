import os
import subprocess

PATH_INPUT = "../results"
PATH_OUTPUT = "../results/mscoco"
RANDOM_STATE = 0

START = 0
STOP = 1000

for model_name, batch_size in {
    "openai/clip-vit-base-patch32": 64,
    "openai/clip-vit-base-patch16": 64,
}.items():
    for mode in [
        'banzhaf', 
        'shapley'
    ]:
        if mode == "banzhaf":
            budget = 2**21
            for p_sampler in [
                0.5,
                0.3,
                0.7,
            ]:
                subprocess.run([
                    "sbatch", 
                    "run_explain_mscoco.sh", 
                    model_name,
                    os.path.join(PATH_INPUT, model_name),
                    os.path.join(PATH_OUTPUT, model_name, mode, str(p_sampler)),
                    str(START),
                    str(STOP),
                    mode,
                    str(p_sampler),
                    str(budget),
                    str(batch_size),
                    str(RANDOM_STATE)
                ])
        elif mode == "shapley":
            # empirical adjustment of budget to equalize computation time
            budget = 2**17
            subprocess.run([
                "sbatch", 
                "run_explain_mscoco.sh", 
                model_name,
                os.path.join(PATH_INPUT, model_name),
                os.path.join(PATH_OUTPUT, model_name, mode),
                str(START),
                str(STOP),
                mode,
                str(0.5),
                str(budget),
                str(batch_size),
                str(RANDOM_STATE)
            ])