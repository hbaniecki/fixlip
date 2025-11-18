import sys
import os

sys.path.append('../')
import src



if __name__ == '__main__':
    ID = 2892 # insertion/deletion: ID: 2892; Subsets: [3, 9, 25, 45, 57]
    METHOD = "shapley" # gradeclip shapley banzhaf_crossmodal
    MODEL_NAME = "openai/clip-vit-base-patch32"
    DATA_PATH = f'../results/mscoco/{MODEL_NAME}/{METHOD}'
    SAVE_DIR = "."
    os.makedirs(SAVE_DIR, exist_ok=True)

    src.plot_ultimate.create_plots_for_instance(
        instance_id=ID,
        data_path=DATA_PATH,
        model_name=MODEL_NAME,
        order=2,
        save_dir=SAVE_DIR,
        verbose=True,
        greedy_clique_sizes=[3, 9, 25, 45, 57],
        debug=False,
        save=True,
        plot_lif=True,
        plot_clique_as_gray=True,
        plot_mif=True,
        plot_heatmap=True,
        font_size_heatmap=22,
        heatmap_figsize=(5.5, 6.25),
        heatmap_margin=0.3,
        color_mask_white=True,
        opacity_white=0.9,
        max_value=2.5
    )