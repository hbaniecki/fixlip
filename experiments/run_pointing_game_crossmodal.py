import os
import subprocess

GAMES = [
    ['goldfish', 'husky', 'pizza', 'tractor'],
    ['cat', 'goldfish', 'plane', 'pizza'],
    ['banana', 'cat', 'tractor', 'ball'],
    ['husky', 'banana', 'plane', 'church'],
    ['pizza', 'ipod', 'goldfish', 'banana'],
    ['ipod', 'cat', 'husky', 'plane'],
    ['tractor', 'ball', 'banana', 'ipod'],
    ['plane', 'church', 'ball', 'goldfish'],
    ['church', 'pizza', 'ipod', 'cat'],
    ['ball', 'husky', 'banana', 'tractor'],
]
PATH_OUTPUT = "../results/imagenet_pointing_game"
RANDOM_STATE = 0
BUDGET = 2**20
MODE = "banzhaf"

for game in GAMES:
    PATH_INPUT = f'../data/imagenet_pointing_game/{"_".join(game)}'
    for model_name, batch_size in {
        "openai/clip-vit-base-patch16": 64,
        "openai/clip-vit-base-patch32": 64,
        "google/siglip2-base-patch32-256": 64,
        "google/siglip2-base-patch16-224": 64,
        "google/siglip2-large-patch16-256": 32,
        "google/siglip-base-patch16-224": 64,
        "google/siglip-large-patch16-256": 32
    }.items():
        for p_sampler in [
            0.5,
            0.3,
            0.7,
        ]:
            for i in range(1, 5):
                class_labels = game[:i]
                if "ipod" in class_labels or "goldfish" in class_labels or "husky" in class_labels:
                    continue
                cl = "_".join(class_labels)
                subprocess.run([
                    "sbatch", 
                    "run_pointing_game_crossmodal.sh", 
                    model_name,
                    PATH_INPUT,
                    os.path.join(PATH_OUTPUT, model_name, f'{MODE}_crossmodal', str(p_sampler), cl),
                    MODE,
                    str(p_sampler),
                    cl,
                    str(BUDGET),
                    str(batch_size),
                    str(RANDOM_STATE)
                ])