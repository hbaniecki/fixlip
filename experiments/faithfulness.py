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

N_EVAL_COALITIONS = 1000  # number of coalitions to evaluate the faithfulness metrics on

# print the settings
print(f"MODEL_NAME: {MODEL_NAME}", flush=True)
print(f"PATH_INPUT: {PATH_INPUT}", flush=True)
print(f"PATH_OUTPUT: {PATH_OUTPUT}", flush=True)
print(f"START: {START}", flush=True)
print(f"STOP: {STOP}", flush=True)
print(f"BATCH_SIZE: {BATCH_SIZE}", flush=True)
print(f"RANDOM_STATE: {RANDOM_STATE}", flush=True)
print(f"N_EVAL_COALITIONS: {N_EVAL_COALITIONS}", flush=True)

import sys
import os
import time

import clip
import torch
torch.set_float32_matmul_precision("high")
from transformers import CLIPProcessor, CLIPModel
import datasets
from shapiq import InteractionValues
import pandas as pd

sys.path.append('../')
import src

start_time = time.time()

import wandb

with wandb.init(project="", name=f'{PATH_OUTPUT}/{MODEL_NAME}/faith', config=args) as run:

    RESULT_DATA: list[dict[str, float]] = []

    df_metadata = pd.read_csv(os.path.join(PATH_OUTPUT, MODEL_NAME, "mscoco_predictions.csv"), index_col=0)
    top_ids = df_metadata.sort_values("logit", ascending=False).iloc[START:STOP, :].index

    dataset = datasets.load_dataset(
        "clip-benchmark/wds_mscoco_captions",
        split="test",
        streaming=True
    )

    model_huggingface = CLIPModel.from_pretrained(MODEL_NAME)
    model_huggingface.to(0)
    processor_huggingface = CLIPProcessor.from_pretrained(MODEL_NAME)

    model_openai, processor_openai = clip.load("ViT-B/32" if MODEL_NAME.endswith("32") else "ViT-B/16", device=1)


    n_iter = 0
    for i, d in enumerate(dataset):
        if i not in top_ids:
            continue

        # load the interaction values only if all are present --------------------------------------
        explanations_huggingface = {}
        explanations_openai = {}

        # banzhaf 0.3  -------------------------------------------------------------------------
        banzhaf_p = "0.3"
        interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "banzhaf", banzhaf_p, f"iv_order1_{i}.pkl")
        banzhaf_1_03 = InteractionValues.load(interaction_path)
        explanations_huggingface["banzhaf_1_03"] = banzhaf_1_03
        interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "banzhaf", banzhaf_p, f"iv_order2_{i}.pkl")
        banzhaf_2_03 = InteractionValues.load(interaction_path)
        explanations_huggingface["banzhaf_2_03"] = banzhaf_2_03

        # banzhaf 0.5  -------------------------------------------------------------------------
        banzhaf_p = "0.5"
        interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "banzhaf", banzhaf_p, f"iv_order1_{i}.pkl")
        banzhaf_1_05 = InteractionValues.load(interaction_path)
        explanations_huggingface["banzhaf_1_05"] = banzhaf_1_05
        interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "banzhaf", banzhaf_p, f"iv_order2_{i}.pkl")
        banzhaf_2_05 = InteractionValues.load(interaction_path)
        explanations_huggingface["banzhaf_2_05"] = banzhaf_2_05

        # banzhaf 0.7  -------------------------------------------------------------------------
        banzhaf_p = "0.7"
        interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "banzhaf", banzhaf_p, f"iv_order1_{i}.pkl")
        banzhaf_1_07 = InteractionValues.load(interaction_path)
        explanations_huggingface["banzhaf_1_07"] = banzhaf_1_07
        interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "banzhaf", banzhaf_p, f"iv_order2_{i}.pkl")
        banzhaf_2_07 = InteractionValues.load(interaction_path)
        explanations_huggingface["banzhaf_2_07"] = banzhaf_2_07

        # shapley ------------------------------------------------------------------------------
        interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "shapley", f"iv_order1_{i}.pkl")
        shapley_1 = InteractionValues.load(interaction_path)
        explanations_huggingface["shapley_1"] = shapley_1
        interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "shapley", f"iv_order2_{i}.pkl")
        shapley_2 = InteractionValues.load(interaction_path)
        explanations_huggingface["shapley_2"] = shapley_2

        # gradeclip --------------------------------------------------------------------------------
        interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "gradeclip", f"iv_order1_{i}.pkl")
        gradeclip_1 = InteractionValues.load(interaction_path)
        explanations_openai["gradeclip_1"] = gradeclip_1

        # game ---------------------------------------------------------------------------------
        interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "game", f"iv_order1_{i}.pkl")
        game_1 = InteractionValues.load(interaction_path)
        explanations_openai["game_1"] = game_1

        try: # oom error for vit-b/16
            # exclip --------------------------------------------------------------------------------
            interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "exclip", f"iv_order2_{i}.pkl")
            exclip_2 = InteractionValues.load(interaction_path)
            explanations_openai["exclip_2"] = exclip_2
        except:
            pass

        # load image/text and create games ----------------------------------------------------------
        n_iter += 1
        print(f'iter: {n_iter}/{STOP - START}', flush=True)
        input_image = d['jpg']
        input_text = d['txt'].split("\n")[df_metadata.loc[i, "best_text_id"].item()]
        game_huggingface = src.game_huggingface.VisionLanguageGame(
            model_huggingface, processor_huggingface,
            input_image=input_image,
            input_text=input_text,
            batch_size=BATCH_SIZE
        )
        game_openai = src.game_openai.CLIPGame(
            model_openai, processor_openai,
            input_image=input_image,
            input_text=input_text,
            batch_size=BATCH_SIZE,
            patch_size=32 if MODEL_NAME.endswith("32") else 16
        )

        # compute the faithfulness metrics for different p -----------------------------------------
        results = src.evaluation.eval_faithfulness_one_game(
            game=game_huggingface,
            explanations=explanations_huggingface,
            sample_p=0.3,
            sample_mode="banzhaf",
            n_eval_coalitions=N_EVAL_COALITIONS,
            instance_id=i
        )
        RESULT_DATA.extend(results)
        results = src.evaluation.eval_faithfulness_one_game(
            game=game_openai,
            explanations=explanations_openai,
            sample_p=0.3,
            sample_mode="banzhaf",
            n_eval_coalitions=N_EVAL_COALITIONS,
            instance_id=i
        )
        RESULT_DATA.extend(results)

        results = src.evaluation.eval_faithfulness_one_game(
            game=game_huggingface,
            explanations=explanations_huggingface,
            sample_p=0.5,
            sample_mode="banzhaf",
            n_eval_coalitions=N_EVAL_COALITIONS,
            instance_id=i
        )
        RESULT_DATA.extend(results)
        results = src.evaluation.eval_faithfulness_one_game(
            game=game_openai,
            explanations=explanations_openai,
            sample_p=0.5,
            sample_mode="banzhaf",
            n_eval_coalitions=N_EVAL_COALITIONS,
            instance_id=i
        )
        RESULT_DATA.extend(results)

        results = src.evaluation.eval_faithfulness_one_game(
            game=game_huggingface,
            explanations=explanations_huggingface,
            sample_p=0.7,
            sample_mode="banzhaf",
            n_eval_coalitions=N_EVAL_COALITIONS,
            instance_id=i
        )
        RESULT_DATA.extend(results)
        results = src.evaluation.eval_faithfulness_one_game(
            game=game_openai,
            explanations=explanations_openai,
            sample_p=0.7,
            sample_mode="banzhaf",
            n_eval_coalitions=N_EVAL_COALITIONS,
            instance_id=i
        )
        RESULT_DATA.extend(results)

        results = src.evaluation.eval_faithfulness_one_game(
            game=game_huggingface,
            explanations=explanations_huggingface,
            sample_mode="shapley",
            n_eval_coalitions=N_EVAL_COALITIONS,
            instance_id=i
        )
        RESULT_DATA.extend(results)
        results = src.evaluation.eval_faithfulness_one_game(
            game=game_openai,
            explanations=explanations_openai,
            sample_mode="shapley",
            n_eval_coalitions=N_EVAL_COALITIONS,
            instance_id=i
        )
        RESULT_DATA.extend(results)

        if n_iter == 1 or n_iter % 5 == 0:
            # store the current results by overwriting a temporary file
            df_results = pd.DataFrame(RESULT_DATA)
            df_results.to_csv(os.path.join(PATH_OUTPUT, MODEL_NAME, f"eval_faithfulness_temp_{N_EVAL_COALITIONS}_{START}_{STOP}.csv"), index=False)

    # save final results ---------------------------------------------------------------------------
    df_results = pd.DataFrame(RESULT_DATA)
    df_results.to_csv(os.path.join(PATH_OUTPUT, MODEL_NAME, f"eval_faithfulness_{N_EVAL_COALITIONS}_{START}_{STOP}.csv"), index=False)

# print the time taken -------------------------------------------------------------------------
elapsed_time = time.time() - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")