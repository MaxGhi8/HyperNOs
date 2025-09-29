"""
Utility to compare multiple losses for two (or more) models across increasing
complexity (e.g. number of trainable parameters, dataset size, iterations).

Requested change: Original convergence-rate utilities removed. Now we plot 6
lines: 3 different losses for Model A and the same 3 losses for Model B.

Design:
    * Different COLORS distinguish the type of loss (e.g. rel L2, phys loss, val loss)
    * Different MARKERS distinguish the model identity (e.g. circle vs square)
    * Each (model, loss) pair is one line with markers.
    * Error bars optional: pass std dev arrays per loss per model (can be None).
    * Legend: one for losses (colors) and one for models (markers) combined into a single legend block.
"""

import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def plot_models_losses(
    x_values: Sequence[float],
    models_losses: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    loss_colors: Optional[Dict[str, str]] = None,
    model_markers: Optional[Dict[str, str]] = None,
    model_linestyles: Optional[Dict[str, str]] = None,
    title: str = "Loss comparison",
    x_label: str = "Number of trainable parameters",
    y_label: str = "Loss value",
    y_scale: str = "log",
    annotate: bool = False,
    save_name: str = "comparison_losses.png",
):
    """Plot multiple losses for multiple models.

    Parameters
    ----------
    x_values : Sequence[float]
        Shared x-axis values (e.g. parameter counts). Length n.
    models_losses : Dict[str, Dict[str, Dict[str, np.ndarray]]]
        Structure: {model_name: {loss_name: {"mean": np.ndarray[n], "std": np.ndarray[n] (optional)}}}
        "std" can be omitted or None for no error bars.
    loss_colors : optional mapping loss_name -> color string.
    model_markers : optional mapping model_name -> marker style.
    model_linestyles : optional mapping model_name -> line style (e.g. '-', '--').
    y_scale : 'linear' or 'log'.
    annotate : if True, add text labels near points for the mean values.
    save_name : filename (inside figures/ directory) for saving.
    """

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Default palettes if none provided
    if loss_colors is None:
        palette = sns.color_palette("tab10")
        # assign deterministically based on sorted loss names encountered
        unique_losses: List[str] = sorted({ln for m in models_losses.values() for ln in m.keys()})
        loss_colors = {ln: palette[i % len(palette)] for i, ln in enumerate(unique_losses)}
    sorted_models = sorted(models_losses.keys())
    if model_markers is None:
        default_markers = ["o", "s", "D", "^", "v", "P", "X"]
        model_markers = {mn: default_markers[i % len(default_markers)] for i, mn in enumerate(sorted_models)}
    if model_linestyles is None:
        # First model solid, second dashed, then alternate
        base_styles = ["-", "--", "-.", ":"]
        model_linestyles = {mn: base_styles[i % len(base_styles)] for i, mn in enumerate(sorted_models)}

    ax = plt.gca()
    x_arr = np.array(x_values)

    # Collect handles for custom legends
    loss_handles = {}
    model_handles = {}

    for model_name, losses in models_losses.items():
        marker = model_markers.get(model_name, "o")
        linestyle = model_linestyles.get(model_name, "-")
        for loss_name, stats in losses.items():
            mean = np.asarray(stats["mean"], dtype=float)
            std = stats.get("std")
            color = loss_colors.get(loss_name, None)

            (line,) = ax.plot(
                x_arr,
                mean,
                label=f"{model_name} - {loss_name}",
                color=color,
                marker=marker,
                markersize=7,
                linewidth=2,
                linestyle=linestyle,
            )
            # Store handles for separate legends
            if loss_name not in loss_handles:
                # proxy artist for loss (color only)
                loss_handles[loss_name] = plt.Line2D([0], [0], color=color, lw=2)
            if model_name not in model_handles:
                # proxy artist for model (marker only, neutral color black)
                model_handles[model_name] = plt.Line2D(
                    [0], [0], color="black", marker=marker, linestyle="", markersize=7
                )

            # Error bars (vertical) if std provided
            # if std is not None:
            #     std = np.asarray(std, dtype=float)
            #     for xi, yi, si in zip(x_arr, mean, std):
            #         if np.isnan(si) or si <= 0:
            #             continue
            #         cap_width = 0.02 * (x_arr.max() - x_arr.min())
            #         ax.plot([xi, xi], [yi - si, yi + si], color=color, linewidth=1.3)
            #         ax.plot([xi - cap_width, xi + cap_width], [yi - si, yi - si], color=color, linewidth=1.3)
            #         ax.plot([xi - cap_width, xi + cap_width], [yi + si, yi + si], color=color, linewidth=1.3)

            if annotate:
                for xi, yi in zip(x_arr, mean):
                    ax.annotate(f"{yi:.3g}", (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    if y_scale == "log":
        ax.set_yscale("log")
        ax.set_xscale("log")

    # Format x-axis ticks (scientific) if large
    def sci_fmt(x, pos):  # noqa: ANN001
        if x == 0:
            return "0"
        return f"{x:.1e}".replace("e+0", "e+").replace("e-0", "e-")
    ax.xaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax.set_xticks(x_arr)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    # Legends: create combined bottom-left positioning without overlap.
    # Place loss legend slightly above model legend.
    loss_legend = ax.legend(
        loss_handles.values(), loss_handles.keys(), title="Loss type", loc="lower left", bbox_to_anchor=(0.0, 0.02)
    )
    ax.add_artist(loss_legend)
    ax.legend(
        model_handles.values(),
        model_handles.keys(),
        title="Model",
        loc="lower left",
        bbox_to_anchor=(0.20, 0.02),
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, save_name)
    plt.savefig(out_path)
    plt.close()
    # Reset style
    plt.style.use("default")
    return out_path


if __name__ == "__main__":
    # Example synthetic data for two models, three losses each
    x_vals = [2e6, 20e6, 250e6]

    models_losses_example = {
        "BAMPNO": {
            "Train (rel. L2)": {"mean": [0.0056, 0.0049, 0.00053], "std": [0.0001, 0.0001, 4.71e-5]},
            "Test (rel. L2)": {"mean": [0.015, 0.0086, 0.0079], "std": [0.0003, 0.0002, 0.00005]},
            "Test (rel. H1)": {"mean": [0.081, 0.037, 0.027], "std": [0.001, 0.0001, 0.0001]},
        },
        "FNO": {
            "Train (rel. L2)": {"mean": [0.0078, 0.0025, 0.0013], "std": [2e-4, 6e-5, 8e-5]},
            "Test (rel. L2)": {"mean": [0.026, 0.025, 0.023], "std": [2e-4, 0.0007, 0.00016]},
            "Test (rel. H1)": {"mean": [0.098, 0.088, 0.083], "std": [0.0015, 0.0019, 0.0012]},
        },
    }

    out_path = plot_models_losses(
        x_values=x_vals,
        models_losses=models_losses_example,
        title="Model loss comparison",
        y_label="Relative error",
        annotate=False,
    )
    print(f"Done. Figure saved to {out_path}")
