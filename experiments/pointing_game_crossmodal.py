import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--model_name', type=str)
parser.add_argument('--path_input', type=str)
parser.add_argument('--path_output', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--p_sampler', type=float)
parser.add_argument('--class_labels', type=str)
parser.add_argument('--budget', type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--random_state', default=0, type=int)
args = parser.parse_args()
MODEL_NAME = args.model_name
PATH_INPUT = args.path_input
PATH_OUTPUT = args.path_output
MODE = args.mode
P_SAMPLER = args.p_sampler
CLASS_LABELS = args.class_labels
BUDGET = args.budget
BATCH_SIZE = args.batch_size
RANDOM_STATE = args.random_state

print(f'-- Input: {PATH_INPUT}', flush=True)
print(f'-- Output: {PATH_OUTPUT}', flush=True)

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'- Device: {DEVICE}', flush=True)
torch.set_float32_matmul_precision("high")
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor

import os
if not os.path.exists(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

import sys
sys.path.append('../')
import src
src.utils.set_seed(RANDOM_STATE)

import wandb
from PIL import Image
import matplotlib.pyplot as plt

with wandb.init(project="", name=PATH_OUTPUT, config=args) as run:

    if "siglip" in MODEL_NAME:
        model = AutoModel.from_pretrained(MODEL_NAME)
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
    elif "clip" in MODEL_NAME:
        model = CLIPModel.from_pretrained(MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    input_text = CLASS_LABELS.replace("_", " ")

    for i in range(50):
        if "siglip2" not in MODEL_NAME and "siglip" in MODEL_NAME and "husky" in input_text:
            break
        input_image = Image.open(os.path.join(PATH_INPUT, f'{i}.jpg'))

        game = src.game_huggingface.VisionLanguageGame(
            model, processor, 
            input_image=input_image,
            input_text=input_text,
            batch_size=BATCH_SIZE
        )
        fixlip = src.fixlip.FIxLIP(
            n_players_image=game.n_players_image, 
            n_players_text=game.n_players_text, 
            mode=MODE,
            max_order=2, 
            p=P_SAMPLER,
            random_state=RANDOM_STATE
        )

        if game.n_players_image == 49 or game.n_players_image == 64:
            top_k = 5 * game.n_players_text
            interaction_lookup = None
        elif game.n_players_image == 196 or game.n_players_image == 256:
            top_k = 20 * game.n_players_text
            interaction_lookup = src.utils.create_crossmodal_interaction_lookup(game.n_players_image, game.n_players_text)

        budget_text = min(64, 2 ** game.n_players_text)
        budget_image = min(2 ** 18, int(BUDGET / budget_text))
        interaction_values = fixlip.approximate_crossmodal(
            game=game, 
            budget_text=budget_text,
            budget_image=budget_image,
            interaction_lookup=interaction_lookup
        )
        interaction_values.save(os.path.join(PATH_OUTPUT, f'iv_order2_{i}.pkl'))

        banzhaf_values = src.utils.convert_iv_to_first_order(interaction_values)
        banzhaf_values.save(os.path.join(PATH_OUTPUT, f'iv_order1_{i}.pkl'))

        ## visualize explanations
        if game.model_type == "siglip":
            text_tokens = input_text.split(" ")
        else:
            text_tokens = game.inputs.tokens()
            if "siglip2" in MODEL_NAME:
                text_tokens = text_tokens[0:game.n_players_text]
                text_tokens = [token.replace('▁', '') for token in text_tokens]
            elif "clip" in MODEL_NAME:
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
            figsize=(6, 6),
            show=False
        ) 
        fig.suptitle(f'{MODEL_NAME} {MODE} {P_SAMPLER if MODE == "banzhaf" else ""}', fontsize=20, y=1.05)
        fig.savefig(os.path.join(PATH_OUTPUT, f'ex_order2_{i}.png'), bbox_inches='tight')
        plt.close(fig)
        fig = src.plot.plot_image_and_text_together(
            img=input_image_denormalized, 
            text=text_tokens, 
            image_players=list(range(game.n_players_image)), 
            iv=banzhaf_values, 
            normalize_jointly=False,
            figsize=(6, 6),
            show=False
        ) 
        fig.suptitle(f'{MODEL_NAME} {MODE} {P_SAMPLER if MODE == "banzhaf" else ""}', fontsize=20, y=1.05)
        fig.savefig(os.path.join(PATH_OUTPUT, f'ex_order1_{i}.png'), bbox_inches='tight')
        plt.close(fig)
        ##
        run.config['budget_actual'] = budget_text * budget_image

wandb.finish()