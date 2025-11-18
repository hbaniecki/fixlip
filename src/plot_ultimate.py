import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapiq import InteractionValues
from tqdm import tqdm
import datasets
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor

from src.clique import get_clique_value, get_cliques_greedy_mif_lif
from src.game_huggingface import VisionLanguageGame
from src.plot import plot_interaction_subset, plot_image_and_text_together, image_torch_to_array
from src.utils import convert_iv_to_first_order


def create_plots_for_instance(
    data_path: str,
    model_name: str,
    *,
    instance_id: int | None = None,
    order: int = 2,
    input_image: np.ndarray | None = None,
    input_text: str | None = None,
    interaction_values: InteractionValues | None = None,
    max_value: float | str | None = None,
    interactions_figsize = (12, 7),
    interactions_image_span = 0.75,
    interactions_margin: float = -1.5,
    interactions_fontsize: int = 30,
    interactions_top_k: int = 100,
    interactions_line_padding: float = 2.0,
    interactions_sort_by_abs: bool = True,
    heatmap_figsize = (9, 10),
    heatmap_margin: float = 0.2,
    heatmap_font_size: int = 32,
    heatmap_line_padding: float = 0.25,
    opacity_white: float = 0.8,
    mif_greedy_clique_sizes: list[int] = None,
    lif_greedy_clique_sizes: list[int] = None,
    condition_on_players: list[int] = None,
    condition_on_normalize_jointly: bool = True,
    cliques_to_plot: list[set[int]] = None,
    color_mask_white: bool = True,
    plot_interactions_plot: bool = False,
    plot_main_effects_in_interactions_plot: bool = True,
    plot_heatmap: bool = False,
    plot_heatmap_normalized_independently: bool = False,
    plot_original_input: bool = False,
    plot_clique_as_gray: bool = False,
    plot_clique_as_clique: bool = False,
    save_dir: str = None,
    debug: bool = False,
    save: bool = False,
    verbose: bool = False,
    add_instance_id_ontop: bool = False
):
    """Creates a set of plots for the given instance_id."""
    # load the interaction values from the DATA_PATH -------------------------------------------
    if interaction_values is None:
        interaction_path = os.path.join(data_path, f"iv_order{order}_{instance_id}.pkl")
        interaction_values = InteractionValues.load(interaction_path)

    if tuple() in interaction_values.interaction_lookup:
        # set the score to almost zero
        interaction_values.values[interaction_values.interaction_lookup[tuple()]] = 1e-10

    if isinstance(max_value, str):
        # get the multiplier from the string
        multiplier = float(max_value)
        max_value_in_interactions = max(abs(interaction_values.values))
        max_value = multiplier * max_value_in_interactions
        if verbose:
            print(f"Max value was set to {max_value} based on the multiplier {multiplier}.")

    # parse model information ----------------------------------------------------------------------
    model_identifier = model_name.split("/")[-1]
    model_name_short = "siglip" if "siglip" in model_name else "clip"
    patch_size = 32 if "patch32" in model_name else 16
    model_name_short += f"-{patch_size}"
    save_name = f"iv_order{order}_{instance_id}_{model_name_short}"

    # get the save directory -----------------------------------------------------------------------
    if save_dir is None:
        this_dir = os.path.dirname(__file__)
        save_dir = "."
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print(interaction_values)

    # get the data and pre-process it ----------------------------------------------------------
    if input_image is not None and input_text is not None:
        if instance_id is None:
            instance_id = "provided_image"
    else:
        if instance_id is None:
            raise ValueError("instance_id is not provided.")
        df_metadata = pd.read_csv(os.path.join("..", "results", model_name, "mscoco_predictions.csv"), index_col=0)
        dataset = datasets.load_dataset(
            "clip-benchmark/wds_mscoco_captions",
            split="test",
        )
        data = dataset[instance_id]
        input_image = data['jpg']
        input_text = data['txt'].split("\n")[df_metadata.loc[instance_id, "best_text_id"].item()]

    # set up the correct game ----------------------------------------------------------------------
    if "siglip" in model_name_short:
        game = VisionLanguageGame(
            model=AutoModel.from_pretrained(model_name),
            processor=AutoProcessor.from_pretrained(model_name),
            input_image=input_image,
            input_text=input_text,
            batch_size=1
        )
        text_tokens = game.inputs.tokens()
        text_tokens = text_tokens[0:game.n_players_text]
        text_tokens = [token.replace('▁', '') for token in text_tokens]
    else:  # model is "clip"
        game = VisionLanguageGame(
            model=CLIPModel.from_pretrained(model_name),
            processor=CLIPProcessor.from_pretrained(model_name),
            input_image=input_image,
            input_text=input_text,
        )
        text_tokens = game.inputs.tokens()
        text_tokens = text_tokens[1:-1]
        text_tokens = [token.replace('</w>', '') for token in text_tokens]

    n_players_image = game.n_players_image
    image_array = image_torch_to_array(
        game.inputs['pixel_values'].squeeze(0),
        game.processor.image_processor.image_mean,
        game.processor.image_processor.image_std
    )

    # plot original input --------------------------------------------------------------------------
    if plot_original_input:
        plot_image_and_text_together(
            img=image_array,
            text=text_tokens,
            image_players=list(range(n_players_image)),
            iv=interaction_values,
            plot_interactions=False,
            normalize_jointly=False,
            color_text=False,
            plot_heatmap=False,
            figsize=heatmap_figsize,
            image_span=heatmap_figsize[0] / heatmap_figsize[1],
            show=False,
            margin=heatmap_margin,
            line_padding=heatmap_line_padding,
            fontsize=heatmap_font_size,
            debug=False
        )
        save_path = os.path.join(save_dir, save_name + f"_original_input.pdf")
        if save:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    # plot the interaction values ------------------------------------------------------------------
    if plot_interactions_plot:
        plot_image_and_text_together(
            img=image_array,
            text=text_tokens,
            image_players=list(range(n_players_image)),
            iv=interaction_values,
            plot_interactions=True,
            top_k=interactions_top_k,
            sort_by_abs=interactions_sort_by_abs,
            normalize_jointly=True,
            figsize=interactions_figsize,
            image_span=interactions_image_span,
            color_text=plot_main_effects_in_interactions_plot,
            plot_heatmap=plot_main_effects_in_interactions_plot,
            max_value=max_value,
            show=False,
            margin=interactions_margin,
            line_padding=interactions_line_padding,
            fontsize=interactions_fontsize,
            debug=debug
        )
    if add_instance_id_ontop:
        # plot the instance id in the top left corner
        plt.text(
            0.02,
            0.98,
            f"ID: {instance_id}\n({model_name_short})",
            fontsize=interactions_fontsize,
            ha='left',
            va='top',
            color="black",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
        )
    save_path = os.path.join(save_dir, save_name + f"_interactions.pdf")
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    # plot heatmap of interaction values -----------------------------------------------------------
    if plot_heatmap:
        iv_first_order = convert_iv_to_first_order(interaction_values)
        plot_image_and_text_together(
            img=image_array,
            text=text_tokens,
            image_players=list(range(n_players_image)),
            iv=iv_first_order,
            plot_interactions=False,
            normalize_jointly=True,
            figsize=heatmap_figsize,
            image_span=heatmap_figsize[0] / heatmap_figsize[1],
            show=False,
            margin=heatmap_margin,
            line_padding=heatmap_line_padding,
            fontsize=heatmap_font_size,
            debug=False
        )
        save_path = os.path.join(save_dir, save_name + f"_heatmap_normalized_jointly.pdf")
        if save:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    if plot_heatmap_normalized_independently:
        iv_first_order = convert_iv_to_first_order(interaction_values)
        plot_image_and_text_together(
            img=image_array,
            text=text_tokens,
            image_players=list(range(n_players_image)),
            iv=iv_first_order,
            plot_interactions=False,
            normalize_jointly=False,
            figsize=heatmap_figsize,
            image_span=heatmap_figsize[0] / heatmap_figsize[1],
            show=False,
            margin=heatmap_margin,
            line_padding=heatmap_line_padding,
            fontsize=heatmap_font_size,
            debug=False
        )
        save_path = os.path.join(save_dir, save_name + f"_heatmap.pdf")
        if save:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    # plot the conditioned interaction values ------------------------------------------------------
    if condition_on_players is not None:
        for player in condition_on_players:
            plot_image_and_text_together(
                img=image_array,
                text=text_tokens,
                image_players=list(range(n_players_image)),
                iv=interaction_values,
                plot_interactions=False,
                normalize_jointly=condition_on_normalize_jointly,
                figsize=heatmap_figsize,
                image_span=heatmap_figsize[0] / heatmap_figsize[1],
                show=False,
                margin=heatmap_margin,
                line_padding=heatmap_line_padding,
                fontsize=heatmap_font_size,
                debug=False,
                condition_on_player=player
            )
            save_path = os.path.join(save_dir, save_name + f"_condition_on_{player}.pdf")
            if save:
                plt.savefig(save_path, bbox_inches='tight')
            plt.show()

    # plot the provided cliques --------------------------------------------------------------------
    if cliques_to_plot is not None:
        for clique in cliques_to_plot:
            clique_score = get_clique_value(interaction_values, clique)
            clique_str = "".join([str(c) for c in sorted(clique)])
            if verbose:
                print(f"clique: {clique}, clique_score: {clique_score}")

            if plot_clique_as_clique:
                plot_interaction_subset(
                    iv=interaction_values,
                    clique=clique,
                    img=image_array,
                    text=text_tokens,
                    image_players=list(range(n_players_image)),
                    plot_main_effect=True
                )
                save_path = os.path.join(save_dir, save_name + f"_clique_{clique_str}.pdf")
                if save:
                    plt.savefig(save_path, bbox_inches='tight')
                plt.show()

            if plot_clique_as_gray:
                plot_image_and_text_together(
                    img=image_array,
                    text=text_tokens,
                    player_mask=clique,
                    image_players=list(range(n_players_image)),
                    iv=interaction_values,
                    color_mask_white=color_mask_white,
                    plot_heatmap=False,
                    plot_interactions=False,
                    color_text=False,
                    figsize=heatmap_figsize,
                    image_span=heatmap_figsize[0] / heatmap_figsize[1],
                    show=False,
                    margin=heatmap_margin,
                    line_padding=heatmap_line_padding,
                    fontsize=heatmap_font_size,
                    opacity_white=opacity_white,
                )
                save_path = os.path.join(save_dir, save_name + f"_clique_{clique_str}_gray.pdf")
                if save:
                    plt.savefig(save_path, bbox_inches='tight')
                plt.show()

    if mif_greedy_clique_sizes is None and lif_greedy_clique_sizes is None:
        return

    # plot the cliques retrieved from the greedy algorithm -----------------------------------------
    if verbose:
        print(f"Computing greedy cliques for {instance_id}:")

    mif_gr, lif_gr = get_cliques_greedy_mif_lif(
        iv=interaction_values,
        start_players=None,
        max_size=None,
        reverse=True,
        verbose=False,
    )

    for i in range(1, mif_gr.shape[0]):
        clique_mif = set(np.where(mif_gr[i])[0])
        clique_lif = set(np.where(lif_gr[i])[0])
        size = len(clique_mif)  # both are the same size

        greedy_cliques_to_plot = {}
        if size not in mif_greedy_clique_sizes and size not in lif_greedy_clique_sizes:
            continue
        if size in mif_greedy_clique_sizes:
            greedy_cliques_to_plot["mif"] = clique_mif
        if size in lif_greedy_clique_sizes:
            greedy_cliques_to_plot["lif"] = clique_lif

        for clique_kind, clique in greedy_cliques_to_plot.items():
            clique_score = get_clique_value(interaction_values, clique)
            if verbose:
                print(f"clique {clique_kind}: {clique}, clique_score: {clique_score}")

            if plot_clique_as_clique:
                plot_interaction_subset(
                    iv=interaction_values,
                    clique=clique,
                    img=image_array,
                    text=text_tokens,
                    image_players=list(range(n_players_image)),
                    plot_main_effect=True
                )
                save_path = os.path.join(save_dir, save_name + f"_{clique_kind}_{size}.pdf")
                if save:
                    plt.savefig(save_path, bbox_inches='tight')
                plt.show()

            if plot_clique_as_gray:
                plot_image_and_text_together(
                    img=image_array,
                    text=text_tokens,
                    player_mask=clique,
                    image_players=list(range(n_players_image)),
                    iv=interaction_values,
                    plot_heatmap=False,
                    color_mask_white=color_mask_white,
                    plot_interactions=False,
                    color_text=False,
                    figsize=heatmap_figsize,
                    image_span=heatmap_figsize[0] / heatmap_figsize[1],
                    show=False,
                    margin=heatmap_margin,
                    line_padding=heatmap_line_padding,
                    fontsize=heatmap_font_size,
                    opacity_white=opacity_white,
                )
                save_path = os.path.join(save_dir, save_name + f"_{clique_kind}_{size}_gray.pdf")
                if save:
                    plt.savefig(save_path, bbox_inches='tight')
                plt.show()


def plot_from_files(
    data_path: str,
    model_name: str,
    order: int = 2,
    instance_override: list = None,
    max_plots: int = 1_000_000_000,
    save: bool = False,
):
    # get the instances ----------------------------------------------------------------------------
    if instance_override is not None and len(instance_override) > 0:
        instance_ids = instance_override
        print(f"Overriding instance_ids to {instance_ids}.")
        print("Instance IDs:")
        print("\n".join([str(idx) for idx in instance_ids]))
    else:
        all_interactions = os.listdir(data_path)
        all_interactions = [f for f in all_interactions if f.endswith(".pkl")]
        instance_ids = [int(f.split("_")[2].split(".")[0]) for f in all_interactions]
        instance_ids = [idx for idx in instance_ids if f"order{order}" in all_interactions[0]]
        instance_ids = sorted(list(set(instance_ids)))
        print(f"Found {len(instance_ids)} instance_ids in {data_path}.")
        print("Instance IDs:")
        print("\n".join([str(idx) for idx in instance_ids]))


    pbar = tqdm(total=len(instance_ids), desc="Plotting interactions")
    for i, idx in enumerate(instance_ids):
        if i >= max_plots:
            break
        try:
            create_plots_for_instance(
                instance_id=idx,
                save=save,
                data_path=data_path,
                model_name=model_name,
                plot_interactions_plot=True
            )
        except Exception as e:
            print(f"ID {idx} caused an error: {e}")
            continue
        pbar.update(1)
