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

from transformers import CLIPProcessor, CLIPModel
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
import matplotlib.pyplot as plt

with wandb.init(project="", name=f'{PATH_OUTPUT}/explain', config=args) as run:

    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    dataset = datasets.load_dataset(
        "clip-benchmark/wds_mscoco_captions",
        split="test",
        streaming=True
    )
    
    size_clique = 72
    run.config.update({"size_clique": size_clique, "start": START, "stop": STOP})

    df_metadata = pd.read_csv(os.path.join(PATH_INPUT, "mscoco_predictions.csv"), index_col=0)
    top_ids = df_metadata.sort_values("logit", ascending=False).iloc[START:STOP, :].index

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
        
        if game.n_players_image == 49:
            top_k = 20
            if MODE == "banzhaf":
                fixlip_attribution = src.fixlip.FIxLIP(
                    n_players_image=game.n_players_image, 
                    n_players_text=game.n_players_text, 
                    mode=MODE,
                    max_order=1, 
                    p=P_SAMPLER,
                    random_state=RANDOM_STATE
                )
                attribution_values = fixlip_attribution.approximate_crossmodal(
                    game=game, 
                    budget=BUDGET
                )
                fixlip_interaction = src.fixlip.FIxLIP(
                    n_players_image=game.n_players_image, 
                    n_players_text=game.n_players_text, 
                    mode=MODE,
                    max_order=2, 
                    p=P_SAMPLER,
                    random_state=RANDOM_STATE
                )
                interaction_values = fixlip_interaction.approximate_crossmodal(
                    game=game, 
                    budget=BUDGET
                )
            elif MODE == "shapley":
                fixlip_attribution = src.fixlip.FIxLIP(
                    n_players=game.n_players, 
                    mode=MODE,
                    max_order=1, 
                    random_state=RANDOM_STATE
                )
                attribution_values = fixlip_attribution.approximate(
                    game=game, 
                    budget=BUDGET
                )
                fixlip_interaction = src.fixlip.FIxLIP(
                    n_players=game.n_players, 
                    mode=MODE,
                    max_order=2, 
                    random_state=RANDOM_STATE
                )
                interaction_values = fixlip_interaction.approximate(
                    game=game, 
                    budget=BUDGET
                )
            # attribution_values = src.utils.convert_iv_to_first_order(interaction_values)
        elif game.n_players_image == 196: # interactive approach
            top_k = 80
            if MODE == "banzhaf":
                fixlip_attribution = src.fixlip.FIxLIP(
                    n_players_image=game.n_players_image, 
                    n_players_text=game.n_players_text, 
                    mode=MODE,
                    max_order=1, 
                    p=P_SAMPLER,
                    random_state=RANDOM_STATE
                )
                attribution_values = fixlip_attribution.approximate_crossmodal(
                    game=game, 
                    budget=BUDGET
                )
                players_clique = src.utils.get_top_clique_players(attribution_values, size_clique, game.n_players_image, game.n_players_text)
                interaction_lookup = src.utils.create_subset_interaction_lookup(game.n_players, players_clique)
                fixlip_interaction = src.fixlip.FIxLIP(
                    n_players_image=game.n_players_image, 
                    n_players_text=game.n_players_text, 
                    mode=MODE,
                    max_order=2, 
                    p=P_SAMPLER,
                    random_state=RANDOM_STATE
                )
                interaction_values = fixlip_interaction.approximate_crossmodal(
                    game=game, 
                    budget=BUDGET,
                    interaction_lookup=interaction_lookup
                )
            elif MODE == "shapley":
                fixlip_attribution = src.fixlip.FIxLIP(
                    n_players=game.n_players, 
                    mode=MODE,
                    max_order=1, 
                    random_state=RANDOM_STATE
                )
                attribution_values = fixlip_attribution.approximate(
                    game=game, 
                    budget=BUDGET
                )
                players_clique = src.utils.get_top_clique_players(attribution_values, size_clique, game.n_players_image, game.n_players_text)
                interaction_lookup = src.utils.create_subset_interaction_lookup(game.n_players, players_clique)
                fixlip_interaction = src.fixlip.FIxLIP(
                    n_players=game.n_players,
                    mode=MODE,
                    max_order=2, 
                    random_state=RANDOM_STATE
                )
                interaction_values = fixlip_interaction.approximate(
                    game=game, 
                    budget=BUDGET,
                    interaction_lookup=interaction_lookup
                )

        attribution_values.save(os.path.join(PATH_OUTPUT, f'iv_order1_{i}.pkl'))
        interaction_values.save(os.path.join(PATH_OUTPUT, f'iv_order2_{i}.pkl'))

        ## visualize explanations
        text_tokens = game.inputs.tokens()
        text_tokens = text_tokens[1:-1]
        text_tokens = [token.replace('</w>', '') for token in text_tokens]
        assert len(text_tokens) == game.n_players_text
        players_text = list(range(game.n_players_image, game.n_players))
        assert game.n_players == interaction_values.n_players == max(players_text) + 1
        input_image_processed = game.inputs['pixel_values'].squeeze(0)
        input_image_denormalized = src.utils.denormalize(
            input_image_processed, 
            game.processor.image_processor.image_mean, 
            game.processor.image_processor.image_std
        ).permute(1, 2, 0).numpy()
        fig = src.plot.plot_image_and_text_together(
            img=input_image_denormalized, 
            text=text_tokens, 
            image_players=list(range(game.n_players_image)), 
            iv=interaction_values, 
            plot_interactions=True,
            top_k=top_k,
            normalize_jointly=True,
            figsize=(8, 8),
            show=False
        ) 
        fig.suptitle(f'{MODEL_NAME} {MODE} {P_SAMPLER if MODE.startswith("banzhaf") else ""}', fontsize=20, y=1.05)
        fig.savefig(os.path.join(PATH_OUTPUT, f'ex_order2_{i}.png'), bbox_inches='tight')
        plt.close(fig)
        fig = src.plot.plot_image_and_text_together(
            img=input_image_denormalized, 
            text=text_tokens, 
            image_players=list(range(game.n_players_image)), 
            iv=attribution_values, 
            normalize_jointly=False,
            figsize=(6, 6),
            show=False
        ) 
        fig.suptitle(f'{MODEL_NAME} {MODE} {P_SAMPLER if MODE.startswith("banzhaf") else ""}', fontsize=20, y=1.05)
        fig.savefig(os.path.join(PATH_OUTPUT, f'ex_order1_{i}.png'), bbox_inches='tight')
        plt.close(fig)
        ##

wandb.finish()