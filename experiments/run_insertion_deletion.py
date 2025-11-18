import os
import subprocess

PATH_INPUT = "../results/mscoco"
PATH_OUTPUT = "../results"
# MODEL_NAME = "openai/clip-vit-base-patch32"
MODEL_NAME = "openai/clip-vit-base-patch16"
EVERY_K = 100
STARTS = list(range(0, 1000, EVERY_K))

for start in STARTS:
    subprocess.run([
        "sbatch", 
        "run_insertion_deletion.sh", 
        MODEL_NAME,
        PATH_INPUT,
        PATH_OUTPUT,
        str(start),
        str(start + EVERY_K)
    ])