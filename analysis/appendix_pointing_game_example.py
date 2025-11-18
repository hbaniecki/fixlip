import sys
import os
from PIL import Image

sys.path.append('../')
import src

if __name__ == '__main__':
    ID = 8
    METHOD = "banzhaf_crossmodal/0.5"

    input_image = Image.open(f'../data/imagenet_pointing_game/banana_cat_tractor_ball/{ID}.jpg')
    game = "banana cat tractor ball"
    for i in range(1, 5):
        input_text = " ".join(game.split(" ")[:i])
        directory = input_text.replace(" ", "_")
        SAVE_DIR = "."
        os.makedirs(SAVE_DIR, exist_ok=True)
        MODEL_NAME = "google/siglip2-base-patch16-224"
        DATA_PATH = f'../results/imagenet_pointing_game/{MODEL_NAME}/{METHOD}/{directory}'

        src.plot_ultimate.create_plots_for_instance(
            instance_id=ID,
            data_path=DATA_PATH,
            model_name=MODEL_NAME,
            input_image=input_image, 
            input_text=input_text,
            order=2, # 1, 2
            save_dir=SAVE_DIR,
            verbose=True,
            debug=False,
            save=True,
            plot_heatmap=True,
            heatmap_font_size=22,
            heatmap_figsize=(5.5, 6.25),
            heatmap_margin=0.3,
            plot_interactions_plot=True,
            plot_main_effects_in_interactions_plot=False,
            max_value=2.5,
            condition_on_players=[196]
        )