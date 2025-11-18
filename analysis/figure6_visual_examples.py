import sys
import os

sys.path.append('../')
import src



if __name__ == '__main__':
    ID = 1626 
    METHOD = "banzhaf/0.5"
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
        greedy_clique_sizes=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        debug=False,
        save=True,
        plot_clique_as_gray=True,
        plot_clique_as_clique=True,
        plot_lif=True,
        plot_mif=True,
        plot_heatmap=True,
        font_size_heatmap=22,
        heatmap_figsize=(5.5, 6.25),
        heatmap_margin=0.3,
        color_mask_white=True,
        opacity_white=0.8
    )

    # 1100 banzhaf/0.5 openai/clip-vit-base-patch32
    # src.plot_ultimate.create_plots_for_instance(
    #     instance_id=ID,
    #     data_path=DATA_PATH,
    #     model_name=MODEL_NAME,
    #     order=2,
    #     save_dir=SAVE_DIR,
    #     verbose=True,
    #     debug=False,
    #     save=True,
    #     plot_heatmap=True,
    #     font_size_heatmap=22,
    #     heatmap_figsize=(5.5, 6.25),
    #     heatmap_margin=0.3,
    #     condition_on_players=[50, 53, 54, 61],
    # )