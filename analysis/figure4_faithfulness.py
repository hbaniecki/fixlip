from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)


def plot_faithfulness(
    df: pd.DataFrame,
    hue: Literal["sample_p"] | None = "sample_p",
    sample_p: float | None = None,
    show_shapley_distribution: bool = False,
    appendix=False,
    rotated: bool = False
) -> None:
    """This function plots the faithfulness of the methods."""
    # STYLING --------------------------------------------------------------------------------------
    # first test of pallete
    custom_pallet_p_samples_one = {
        0.3: "#94ECBE",
        0.5: "#4A7856",
        0.7: "#345830",
        "Shapley": "#1E3F20",
    }
    # second test of pallete
    custom_pallet_p_samples_two = {
        0.3: "#FBBC04",  # "#00b4d8",
        0.5: "#E37400",  # "#ef27a6",
        0.7: "#EA4335",  # "#ff6f00",
        "Shapley": "#4285F4",  # "#ffbe0b"
    }
    # third test with a shade of gray
    custom_pallet_p_samples_three = {
        0.3: "#dadedf",
        0.5: "#a3adb1",
        0.7: "#6b7b82",
        "Shapley": "#a3adb1",
    }

    custom_pallet_p_samples = custom_pallet_p_samples_two

    custom_palette_methods = {
        "banzhaf_1_03": "#FBBC04",
        "banzhaf_2_03": "#FBBC04",
        "banzhaf_1_05": "#E37400",
        "banzhaf_2_05": "#E37400",
        "banzhaf_1_07": "#EA4335",
        "banzhaf_2_07": "#EA4335",
        "shapley_1": "#4285F4",
        "shapley_2": "#4285F4",
        "game_1": "#9AA0A6",
        "gradeclip_1": "#34A853",
        "exclip_2": "#CC8899",
    }

    styling_params = {
        "palette": "Set2",
        "linewidth": 0.75,
        "saturation": 1.0,
    }

    styling_params_boxplot = {
        "showfliers": False,
    }

    method_names = {
        "banzhaf_1_03": "Banzhaf\nvalues\n$p = 0.3$",
        "banzhaf_2_03": "FIxLIP\n(WBI$_{p=0.3}$)",
        "banzhaf_1_05": "Banzhaf\nvalues",
        "banzhaf_2_05": "FIxLIP\n(WBI$_{p=0.5}$)",
        "banzhaf_1_07": "Banzhaf\nvalues\n$p = 0.7$",
        "banzhaf_2_07": "FIxLIP\n(WBI$_{p=0.7}$)",
        "shapley_1": "Shapley\nvalues",
        "shapley_2": "FIxLIP\n(SI)",
        "game_1": "GAME",
        "gradeclip_1": "Grad-ECLIP",
        "exclip_2": "exCLIP",
    }
    if rotated:
        method_names["banzhaf_2_03"] = "FIxLIP (WBI $p = 0.3$)"
        method_names["banzhaf_2_05"] = "FIxLIP (WBI $p = 0.5$)"
        method_names["banzhaf_2_07"] = "FIxLIP (WBI $p = 0.7$)"
        method_names = {
            k: v.replace("\n", " ") for k, v in method_names.items()
        }


    metric_names = {
        "r2": r"$R^2$",
        "r2_banzhaf": r"$R^2$ (Banzhaf)",
        "spearman_correlation": "$p$-faithfulness correlation",
        "correlation": "Pearson correlation",
        "kendall_tau": "Kendall's tau",
        "cosine_similarity": "Cosine similarity",
    }

    label_title = "$p$-faithfulness\nmetric distribution"

    r2_methods = [
        "shapley_1",
        "banzhaf_1_05",
        "shapley_2",
        "banzhaf_2_03",
        "banzhaf_2_05",
        "banzhaf_2_07"
    ]

    correlation_methods = [
        "exclip_2",
        "game_1",
        "gradeclip_1",
        "shapley_1",
        "banzhaf_1_05",
        "shapley_2",
        "banzhaf_2_03",
        "banzhaf_2_05",
        "banzhaf_2_07"
    ]

    # file handling --------------------------------------------------------------------------------
    save_dir = "."
    if appendix:
        save_dir = os.path.join(save_dir, "appendix")
    os.makedirs(save_dir, exist_ok=True)

    # prepare the data -----------------------------------------------------------------------------
    df.loc[df["sample_mode"] == "shapley", "sample_p"] = "Shapley"
    if sample_p is not None:
        df = df[df["sample_p"] == sample_p]

    # adjust styling parameters --------------------------------------------------------------------
    if hue == "sample_p":
        styling_params["palette"] = custom_pallet_p_samples
        styling_params_boxplot["legend"] = True
        styling_params["gap"] = 0.2
        styling_params["hue_order"] = [0.3, 0.5, 0.7, "Shapley"]
        if not show_shapley_distribution:
            # remove the shapley methods
            df = df[df["sample_p"] != "Shapley"]
            if "Shapley" in styling_params["palette"]:
                styling_params["palette"].pop("Shapley")
            del styling_params["hue_order"][-1]
    else:
        hue = None
        styling_params["palette"] = custom_palette_methods
        styling_params_boxplot["legend"] = False

    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "white"})
    sns.set_context("paper", rc={
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
    })

    # plot correlation -----------------------------------------------------------------------------
    correlation_methods = [m for m in correlation_methods if m in df["method_name"].unique()]
    metrics = ["r2", "spearman_correlation"]
    for metric_to_plot in metrics:
        methods = correlation_methods
        metric_limits = (0, 1)
        if metric_to_plot == "r2":
            methods = r2_methods
        metric_name = metric_names.get(metric_to_plot, metric_to_plot)
        label_names = [method_names[m] for m in methods]
        if rotated:
            fig, ax = plt.subplots(figsize=(6, 3))
            if hue is not None:
                fig, ax = plt.subplots(figsize=(6, len(methods) * 0.75))
            elif metric_to_plot == "r2":
                fig, ax = plt.subplots(figsize=(6, 2.25))
        else:
            fig, ax = plt.subplots(figsize=(len(methods), 3))

        x, y = "method_name", metric_to_plot
        sns.boxplot(
            x=x if not rotated else y,
            y=y if not rotated else x,
            data=df,
            ax=ax,
            hue=hue,
            order=methods,
            **styling_params,
            **styling_params_boxplot
        )

        if appendix:
            title = "CLIP (ViT-B/16)" if large_model else "CLIP (ViT-B/32)"
            ax.set_title(title)
        metric_label = metric_name + f" ($p={sample_p}$)" if sample_p is not None else metric_name
        metric_label += r" $\rightarrow$"
        if not rotated:
            ax.set_ylabel(metric_label)
            ax.set_xlabel("")
            ax.set_ylim(metric_limits)
            ax.set_xticklabels(label_names)
            for i in range(0, len(methods), 2):
                ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.05)
            ax.set_xlim(-0.5, len(methods) - 0.5)
        else:
            # add "-->" to the x-axis labels
            ax.set_xlim(metric_limits)
            ax.set_xlabel(metric_label)
            ax.set_ylabel("")
            ax.set_yticklabels(label_names)
            for i in range(0, len(methods), 2):
                ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.05)
            ax.set_yticks(range(len(methods)))

        if hue == "sample_p":
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [f"$p = {float(l):.1f}$" if l.replace(".", "", 1).isdigit() else l for l in labels]
            if rotated:
                ncol = 1
            else:
                ncol = len(new_labels)
                label_title = label_title.replace("\n", " ")
            ax.legend(handles=handles, labels=new_labels, title=label_title, ncol=ncol, loc="best")
            ax.legend_.get_title().set_fontweight('bold')
        plt.tight_layout(pad=0.05)
        save_name = f"faithfulness_{MODEL_NAME}_{metric_to_plot}"
        if hue is None:
            save_name += "_methods"
        if sample_p is not None:
            save_name += f"_{sample_p}"
        if show_shapley_distribution:
            save_name += "_shapley"
        if rotated:
            save_name += "_rotated"
        plt.savefig(os.path.join(save_dir, save_name + ".pdf"))
        plt.show()


if __name__ == '__main__':

    large_model = True
    if large_model:
        folder = os.path.join("..", "results", "openai", "clip-vit-base-patch16")
        MODEL_NAME = "clip-16"
    else:
        folder = os.path.join("..", "results", "openai", "clip-vit-base-patch32")
        MODEL_NAME = "clip-32"

    data = pd.read_csv(os.path.join(folder, "eval_faithfulness_1000_0_1000.csv"))

    # print unique method names
    print(data["method_name"].unique())

    plot_faithfulness(data, hue=None, sample_p=0.5, appendix=False, rotated=True)
    plot_faithfulness(data, hue=None, sample_p=0.7, appendix=False, rotated=True)

    plot_faithfulness(data, show_shapley_distribution=False, appendix=True, rotated=False)
    plot_faithfulness(data, show_shapley_distribution=False, appendix=True, rotated=True)

