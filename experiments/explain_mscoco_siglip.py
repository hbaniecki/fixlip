import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--model_name', type=str)
parser.add_argument('--path_input', type=str)
parser.add_argument('--path_output', type=str)
parser.add_argument('--start', type=int)
parser.add_argument('--stop', type=int)
parser.add_argument('--mode', type=str)
parser.add_argument('--p_sampler', default=0.5, type=float)
parser.add_argument('--budget', type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--random_state', default=0, type=int)
args = parser.parse_args()
MODEL_NAME = args.model_name
PATH_INPUT = args.path_input
PATH_OUTPUT = args.path_output
START = args.start
STOP = args.stop
MODE = args.mode
P_SAMPLER = args.p_sampler
BUDGET = args.budget
BATCH_SIZE = args.batch_size
RANDOM_STATE = args.random_state

print(f'-- Input: MS COCO', flush=True)
print(f'-- Output: {PATH_OUTPUT}', flush=True)

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'- Device: {DEVICE}', flush=True)
torch.set_float32_matmul_precision("high")

from transformers import AutoProcessor, AutoModel
import datasets
import pandas as pd

import os
if not os.path.exists(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

import sys
sys.path.append('../')
import src
src.utils.set_seed(RANDOM_STATE)

import wandb
import time


with wandb.init(project="", name=f'{PATH_OUTPUT}/explain', config=args) as run:

    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    dataset = datasets.load_dataset(
        "clip-benchmark/wds_mscoco_captions",
        split="test",
        streaming=True
    )

    df_metadata = pd.read_csv(os.path.join(PATH_INPUT, "mscoco_predictions.csv"), index_col=0)
    top_ids = df_metadata.sort_values("logit", ascending=False).iloc[START:STOP, :].index

    result = {'id': [], 'time_explanation': [], 'time_game': []}

    n_iter = 0
    for i, d in enumerate(dataset):
        if i not in top_ids:
            continue
        n_iter += 1
        print(f'iter: {START + n_iter}/{STOP}', flush=True)

        input_image = d['jpg']
        input_text = d['txt'].split("\n")[df_metadata.loc[i, "best_text_id"].item()]
        game = src.game_huggingface.VisionLanguageGame(
            model, processor, 
            input_image=input_image,
            input_text=input_text,
            batch_size=BATCH_SIZE
        )
        
        if MODE == "fixlip":
            fixlip = src.fixlip.FIxLIP(
                n_players_image=game.n_players_image, 
                n_players_text=game.n_players_text, 
                mode="banzhaf",
                max_order=2, 
                p=P_SAMPLER,
                random_state=RANDOM_STATE
            )
            time_explanation_start = time.time()
            interaction_values = fixlip.approximate_crossmodal(
                game=game, 
                budget=BUDGET,
                time_game=True
            )
            time_explanation_end = time.time()
        else:
            fixlip = src.fixlip.FIxLIP(
                n_players=game.n_players, 
                mode=MODE,
                max_order=2, 
                p=P_SAMPLER,
                random_state=RANDOM_STATE
            )
            time_explanation_start = time.time()
            interaction_values = fixlip.approximate(
                game=game, 
                budget=BUDGET,
                time_game=True
            )
            time_explanation_end = time.time()

        # attribution_values = src.utils.convert_iv_to_first_order(interaction_values, p_sampler=P_SAMPLER)
        # attribution_values.save(os.path.join(PATH_OUTPUT, f'iv_order1_{i}.pkl'))
        interaction_values.save(os.path.join(PATH_OUTPUT, f'iv_order2_{i}.pkl'))

        result['id'].append(i)
        result['time_explanation'].append(time_explanation_end - time_explanation_start)
        result['time_game'].append(fixlip.time_game_end - fixlip.time_game_start)

        pd.DataFrame(result).to_csv(os.path.join(PATH_OUTPUT, 'time.csv'), index=False)

wandb.finish()