import sys
import os
import warnings

sys.path.append('../')
import src


def plot_all(
    models,
    data_paths,
    model_name_to_plot_from: str,
):

    all_files = {}
    for model_name, data_path in zip(models, data_paths):
        try:
            files = os.listdir(data_path)
        except FileNotFoundError:
            files = []
        files = [f for f in files if f.endswith(".pkl")]
        files = [f for f in files if "order2" in f]
        all_files[model_name] = files

    files_to_plot = all_files[model_name_to_plot_from]
    for file in files_to_plot:
        # get instance_id
        instance_id = int(file.split("_")[-1].split(".")[0])
        for model_name, data_path in zip(models, data_paths):
            try:
                src.plot_ultimate.create_plots_for_instance(
                    instance_id=instance_id,
                    data_path=data_path,
                    model_name=model_name,
                    order=2,
                    # select plots with params below
                    plot_interactions_plot=True,
                    plot_heatmap_normalized_independently=False,
                    # normalization values:
                    # change the settings of the plots below
                    plot_clique_as_gray=False,
                    plot_clique_as_clique=False,
                    interactions_figsize=(11, 8),
                    interactions_margin=-0.6,
                    interactions_line_padding=1.5,
                    interactions_top_k=30,
                    interactions_image_span=0.83,
                    interactions_fontsize=26,
                    heatmap_figsize=(9, 10.7),
                    heatmap_margin=0.5,
                    heatmap_font_size=35,
                    # save or no save below
                    save=True,
                    verbose=False,
                    add_instance_id_ontop=True,
                    save_dir=".",
                )
            except FileNotFoundError:
                warnings.warn(f"File not found for {model_name}. Skipping...")
                continue
            except Exception as e:
                warnings.warn(f"Failed to plot for {model_name} with error: {e}")
                continue
    sys.exit(0)


if __name__ == '__main__':

    interaction_index = "banzhaf"  # "banzhaf

    models = [
        "openai/clip-vit-base-patch32",
        # "openai/clip-vit-base-patch16",
        # "google/siglip2-base-patch32-256",
    ]
    data_paths = [f"../results/{model}" for model in models]
    if interaction_index == "banzhaf":
        data_paths = [f"{data_path}/banzhaf/0.7" for data_path in data_paths]
    elif interaction_index == "shapley":
        data_paths = [f"{data_path}/shapley" for data_path in data_paths]
    print(f"Plotting for models: {models} and corresponding data paths: {data_paths}")

    # crawl all:
    # plot_all(models, data_paths, model_name_to_plot_from="openai/clip-vit-base-patch32")

    # settings (default settings in braces) --------------------------------------------------------
    interactions_fontsize = 25  # 26
    interactions_margin = -0.6  # -0.6
    interactions_line_padding = 1.5  # 1.5
    interactions_image_span = 0.83  # 0.83
    interactions_figsize = (11, 8)  # (11, 8)
    interactions_sort_by_abs = True  # True
    heatmap_font_size = 35  # 35
    heatmap_margin = 0.5  # 0.5
    heatmap_figsize = (9, 10.7)  # (9, 10.7)
    max_values = {name: None for name in models}  # all None
    max_values["openai/clip-vit-base-patch32"] = "1.05"
    max_values["openai/clip-vit-base-patch16"] = "1.05"
    max_values["google/siglip2-base-patch32-256"] = None
    condition_on_players = {name: None for name in models}
    condition_on_players["openai/clip-vit-base-patch32"] = None  # [50, 51, 56, 59]
    condition_on_players["openai/clip-vit-base-patch16"] = None  # [197, 198, 203, 206]
    condition_on_normalize_jointly = {name: True for name in models}
    condition_on_normalize_jointly["openai/clip-vit-base-patch32"] = False
    condition_on_normalize_jointly["openai/clip-vit-base-patch16"] = False
    interactions_top_k = {name: 20 for name in models}
    interactions_top_k["openai/clip-vit-base-patch32"] = 10
    interactions_top_k["openai/clip-vit-base-patch16"] = 40

    # plot an instance for each model --------------------------------------------------------------
    for model_name, data_path in zip(models, data_paths):
        try:
            src.plot_ultimate.create_plots_for_instance(
                instance_id=4641,
                data_path=data_path,
                model_name=model_name,
                order=2,
                # select plots with params below
                plot_original_input=False,
                plot_interactions_plot=True,
                condition_on_players=condition_on_players[model_name],
                condition_on_normalize_jointly=condition_on_normalize_jointly[model_name],
                plot_heatmap=False,
                plot_heatmap_normalized_independently=False,
                # normalization values:
                max_value=max_values[model_name],
                # change the settings of the plots below
                plot_clique_as_gray=False,
                plot_clique_as_clique=False,
                interactions_figsize=interactions_figsize,
                interactions_margin=interactions_margin,
                interactions_sort_by_abs=interactions_sort_by_abs,
                interactions_line_padding=interactions_line_padding,
                interactions_top_k=interactions_top_k[model_name],
                interactions_image_span=interactions_image_span,
                interactions_fontsize=interactions_fontsize,
                heatmap_figsize=heatmap_figsize,
                heatmap_margin=heatmap_margin,
                heatmap_font_size=heatmap_font_size,
                # save or no save below
                save=True,
                verbose=True,
                debug=False,
            )
        except FileNotFoundError:
            warnings.warn(f"File not found for {model_name}. Skipping...")
            continue
        except Exception as e:
            warnings.warn(f"Failed to plot for {model_name} with error: {e}")
            continue
