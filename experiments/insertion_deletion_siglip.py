import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--model_name', type=str)
parser.add_argument('--path_input', type=str)
parser.add_argument('--path_output', type=str)
parser.add_argument('--path_metadata', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--p_sampler', default=0.5, type=float)
parser.add_argument('--budget', type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--random_state', default=0, type=int)
args = parser.parse_args()
MODEL_NAME = args.model_name
PATH_INPUT = args.path_input
PATH_OUTPUT = args.path_output
PATH_METADATA = args.path_metadata
MODE = args.mode
P_SAMPLER = args.p_sampler
BUDGET = args.budget
BATCH_SIZE = args.batch_size
RANDOM_STATE = args.random_state

print(f'-- Input: {PATH_INPUT}', flush=True)
print(f'-- Output: {PATH_OUTPUT}', flush=True)
print(f'-- Metadata: {PATH_METADATA}', flush=True)

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'- Device: {DEVICE}', flush=True)
torch.set_float32_matmul_precision("high")

from transformers import AutoProcessor, AutoModel
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


with wandb.init(project="", name=f'{PATH_OUTPUT}/aid', config=args) as run:

    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    dataset = datasets.load_dataset(
        "clip-benchmark/wds_mscoco_captions",
        split="test",
        streaming=True
    )

    START, STOP = 900, 1000

    df_metadata = pd.read_csv(os.path.join(PATH_METADATA, "mscoco_predictions.csv"), index_col=0)
    top_ids = df_metadata.sort_values("logit", ascending=False).iloc[START:STOP, :].index

    results = pd.DataFrame({
        'input': [], 
        'mode': [], 
        'p_sampler': [], 
        'budget': [], 
        'order': [],
        'mean': [],
        'mean_normalized': []
    })

    results_details_order1 = {}
    results_details_order2 = {}

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

        path_file = os.path.join(PATH_INPUT, f'iv_order2_{i}.pkl')
        iv_object = shapiq.InteractionValues.load(path_file)

        for order in [1, 2]:
            if order == 1:
                first_order = src.utils.convert_iv_to_first_order(iv_object, p_sampler=P_SAMPLER)
                attribution_values = first_order.get_n_order(1).values
                attribution_values_sorted = np.sort(attribution_values)
                # insertion / deletion, most important first / least important first
                coalition_matrix_deletion_mif = np.stack([attribution_values <= v for v in attribution_values_sorted[::-1]] + [game.empty_coalition])
                predictions_deletion_mif = game.value_function(coalition_matrix_deletion_mif)
                coalition_matrix_deletion_lif = np.stack([attribution_values >= v for v in attribution_values_sorted] + [game.empty_coalition])
                predictions_deletion_lif = game.value_function(coalition_matrix_deletion_lif)

                results_details_order1[i] = {
                    'predictions_deletion_mif': predictions_deletion_mif,
                    'predictions_deletion_lif': predictions_deletion_lif,
                }
            elif order == 2:
                coalition_matrix_deletion_mif, coalition_matrix_deletion_lif = src.clique.get_cliques_greedy_mif_lif(iv=iv_object)
                predictions_deletion_mif = game.value_function(np.concatenate((coalition_matrix_deletion_mif, [game.empty_coalition]), axis=0))
                predictions_deletion_lif = game.value_function(np.concatenate((coalition_matrix_deletion_lif, [game.empty_coalition]), axis=0))

                results_details_order2[i] = {
                    'predictions_deletion_mif': predictions_deletion_mif,
                    'predictions_deletion_lif': predictions_deletion_lif,
                }
            
            assert predictions_deletion_mif[-1] == predictions_deletion_lif[-1]
            assert predictions_deletion_mif[0] == predictions_deletion_lif[0]

            min_value = predictions_deletion_mif[-1]
            max_value = predictions_deletion_mif[0]
            predictions_deletion_mif_normalized = (predictions_deletion_mif - min_value) / (max_value - min_value)
            predictions_deletion_lif_normalized = (predictions_deletion_lif - min_value) / (max_value - min_value)

            results = pd.concat([results, pd.DataFrame({
                'input': [i], 
                'mode': [MODE], 
                'p_sampler': [P_SAMPLER],
                'budget': [BUDGET],
                'order': [order],
                'mean': [np.mean(predictions_deletion_lif - predictions_deletion_mif)],
                'mean_normalized': [np.mean(predictions_deletion_lif_normalized - predictions_deletion_mif_normalized)]
            })])

        if n_iter == 1 or n_iter % 10 == 0:
            results.to_csv(os.path.join(PATH_OUTPUT, f'mscoco_aid_fixlip.csv'), index=False)
            np.save(os.path.join(PATH_OUTPUT, f'mscoco_aid_fixlip.npy'), results_details_order2)
            np.save(os.path.join(PATH_OUTPUT, f'mscoco_aid_fixlip_order1.npy'), results_details_order1)

wandb.finish()