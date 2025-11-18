import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--model_name', type=str)
parser.add_argument('--path_input', type=str)
parser.add_argument('--path_output', type=str)
parser.add_argument('--start', type=int)
parser.add_argument('--stop', type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--random_state', default=0, type=int)
args = parser.parse_args()
MODEL_NAME = args.model_name
PATH_INPUT = args.path_input
PATH_OUTPUT = args.path_output
START = args.start
STOP = args.stop
BATCH_SIZE = args.batch_size
RANDOM_STATE = args.random_state

print(f'-- Input: {PATH_INPUT}/{MODEL_NAME}', flush=True)
print(f'-- Output: {PATH_OUTPUT}', flush=True)

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'- Device: {DEVICE}', flush=True)
torch.set_float32_matmul_precision("high")

from transformers import CLIPProcessor, CLIPModel
import shapiq
import datasets
import numpy as np
import pandas as pd

import os
if not os.path.exists(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

import sys
sys.path.append('../')
import src
src.utils.set_seed(RANDOM_STATE)

import wandb


with wandb.init(project="", name=f'{PATH_OUTPUT}/{MODEL_NAME}/aid', config=args) as run:

    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    dataset = datasets.load_dataset(
        "clip-benchmark/wds_mscoco_captions",
        split="test",
        streaming=True
    )

    orders = [1, 2]
    df_metadata = pd.read_csv(os.path.join(PATH_OUTPUT, MODEL_NAME, "mscoco_predictions.csv"), index_col=0)
    top_ids = df_metadata.sort_values("logit", ascending=False).iloc[START:STOP, :].index

    results_details = {}
    for mode in ['shapley', 'banzhaf/0.3', 'banzhaf/0.5', 'banzhaf/0.7']:
        for order in orders:
            results_details[f'{mode}/order{order}'] = {}

    results = pd.DataFrame({
        'input': [], 
        'mode': [], 
        'order': [],
        'mean': [],
        'mean_normalized': []
    })

    n_iter = 0
    for i, d in enumerate(dataset):
        if i not in top_ids:
            continue
        n_iter += 1
        if n_iter == 1 or n_iter % 5 == 0:
            print(f'iter: {START + n_iter}/{STOP}', flush=True)

        input_image = d['jpg']
        input_text = d['txt'].split("\n")[df_metadata.loc[i, "best_text_id"].item()]
        game = src.game_huggingface.VisionLanguageGame(
            model, processor, 
            input_image=input_image,
            input_text=input_text,
            batch_size=BATCH_SIZE
        )

        for mode in ['shapley', 'banzhaf/0.3', 'banzhaf/0.5', 'banzhaf/0.7']:
            for order in orders:
                path_file = os.path.join(PATH_INPUT, MODEL_NAME, mode, f'iv_order{order}_{i}.pkl')
                iv_object = shapiq.InteractionValues.load(path_file)

                if order == 1:
                    attribution_values = iv_object.get_n_order(1).values
                    attribution_values_sorted = np.sort(attribution_values)
                    # insertion / deletion, most important first / least important first
                    coalition_matrix_deletion_mif = np.stack([attribution_values <= v for v in attribution_values_sorted[::-1]] + [game.empty_coalition])
                    predictions_deletion_mif = game.value_function(coalition_matrix_deletion_mif)
                    coalition_matrix_deletion_lif = np.stack([attribution_values >= v for v in attribution_values_sorted] + [game.empty_coalition])
                    predictions_deletion_lif = game.value_function(coalition_matrix_deletion_lif)

                    predictions_deletion_mif_baseline = predictions_deletion_mif
                    predictions_deletion_lif_baseline = predictions_deletion_lif
                elif order == 2:
                    p_sampler = mode.split("/")
                    if len(p_sampler) == 1:
                        p_sampler = 0.5
                    else:
                        p_sampler = float(p_sampler[1])
                    # check a baseline
                    attribution_values = src.utils.convert_iv_to_first_order(iv_object, p_sampler=p_sampler).get_n_order(1).values
                    attribution_values_sorted = np.sort(attribution_values)
                    # insertion / deletion, most important first / least important first
                    coalition_matrix_deletion_mif_baseline = np.stack([attribution_values <= v for v in attribution_values_sorted[::-1]] + [game.empty_coalition])
                    predictions_deletion_mif_baseline = game.value_function(coalition_matrix_deletion_mif_baseline)
                    coalition_matrix_deletion_lif_baseline = np.stack([attribution_values >= v for v in attribution_values_sorted] + [game.empty_coalition])
                    predictions_deletion_lif_baseline = game.value_function(coalition_matrix_deletion_lif_baseline)
                    # try to find a clique
                    if game.n_players > 100: # vit-b/16
                        start_players = src.clique.get_interesting_starting_players(
                            attribution_values=attribution_values, 
                            first_order_values=iv_object.get_n_order(1).values,
                            k=19
                        )
                        coalition_matrix_deletion_mif, coalition_matrix_deletion_lif = src.clique.get_cliques_greedy_mif_lif(iv=iv_object, start_players=start_players)
                        predictions_deletion_mif = game.value_function(np.concatenate((coalition_matrix_deletion_mif, [game.empty_coalition]), axis=0))
                        predictions_deletion_lif = game.value_function(np.concatenate((coalition_matrix_deletion_lif, [game.empty_coalition]), axis=0))
                    else:
                        coalition_matrix_deletion_mif, coalition_matrix_deletion_lif = src.clique.get_cliques_greedy_mif_lif(iv=iv_object)
                        predictions_deletion_mif = game.value_function(np.concatenate((coalition_matrix_deletion_mif, [game.empty_coalition]), axis=0))
                        predictions_deletion_lif = game.value_function(np.concatenate((coalition_matrix_deletion_lif, [game.empty_coalition]), axis=0))

                results_details[f'{mode}/order{order}'][i] = {
                    'predictions_deletion_mif': predictions_deletion_mif,
                    'predictions_deletion_lif': predictions_deletion_lif,
                    'predictions_deletion_mif_baseline': predictions_deletion_mif_baseline,
                    'predictions_deletion_lif_baseline': predictions_deletion_lif_baseline
                }
                
                assert predictions_deletion_mif[-1] == predictions_deletion_lif[-1]
                assert predictions_deletion_mif[0] == predictions_deletion_lif[0]

                # normalize the curve
                min_value = predictions_deletion_mif[-1]
                max_value = predictions_deletion_mif[0]

                predictions_deletion_mif_normalized = (predictions_deletion_mif - min_value) / (max_value - min_value)
                predictions_deletion_lif_normalized = (predictions_deletion_lif - min_value) / (max_value - min_value)

                predictions_deletion_mif_baseline_normalized = (predictions_deletion_mif_baseline - min_value) / (max_value - min_value)
                predictions_deletion_lif_baseline_normalized = (predictions_deletion_lif_baseline - min_value) / (max_value - min_value)

                results = pd.concat([results, pd.DataFrame({
                    'input': [i], 
                    'mode': [mode], 
                    'order': [order],
                    'mean_normalized': [np.mean(predictions_deletion_lif_normalized - predictions_deletion_mif_normalized)],
                    'mean_normalized_baseline': [np.mean(predictions_deletion_lif_baseline_normalized - predictions_deletion_mif_baseline_normalized)],
                })])

        if n_iter == 1 or n_iter % 5 == 0:
            results.to_csv(os.path.join(PATH_OUTPUT, MODEL_NAME, f'mscoco_aid_fixlip_{START}_{STOP}.csv'), index=False)
            np.save(os.path.join(PATH_OUTPUT, MODEL_NAME, f'mscoco_aid_fixlip_{START}_{STOP}.npy'), results_details)

wandb.finish()