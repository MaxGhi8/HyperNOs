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
from typing import Dict, List, Optional, Sequence, Union

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
    time_annotations: Optional[Dict[str, Sequence[float]]] = None,
    time_fmt: str = "{:.1f}h",
    loss_legend_loc: str = "lower left",
    loss_legend_bbox: Optional[Sequence[float]] = (0.0, 0.02),
    bottom_ylim: Optional[Union[float, int]] = None,
    show_convergence_reference: bool = False,
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
    time_annotations : optional mapping model_name -> Sequence[float] of length len(x_values);
        each value will be formatted with `time_fmt` and placed just above the highest dot
        (across losses) for that model at the corresponding x.
    time_fmt : format string for time annotations, e.g. "{:.1f}h" or "{:.0f}s".
    show_convergence_reference : if True, add a reference line showing x^(-0.5) convergence rate.
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

    # Track, for each model and each x, the maximum y across its losses (to place time labels above top dot)
    model_ymax_per_x: Dict[str, np.ndarray] = {}

    for model_name, losses in models_losses.items():
        marker = model_markers.get(model_name, "o")
        linestyle = model_linestyles.get(model_name, "-")

        # Pre-compute ymax across losses for this model
        means_stack = []
        for loss_name, stats in losses.items():
            mean = np.asarray(stats["mean"], dtype=float)
            means_stack.append(mean)
        if len(means_stack) > 0:
            model_ymax_per_x[model_name] = np.nanmin(np.vstack(means_stack), axis=0)
        else:
            model_ymax_per_x[model_name] = np.full_like(x_arr, np.nan, dtype=float)

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

            # Error bars (vertical) if std provided (kept disabled by default)
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

    # Add convergence reference line if requested
    convergence_ref_handle = None
    if show_convergence_reference:
        # Create reference line: y = C * x^(-0.5)
        # Scale the constant C to fit nicely with the data
        x_ref = np.array(x_values)
        
        # Compute a scaling constant based on the data range
        # Use the geometric mean of all plotted values to position the line
        all_means = []
        for model_losses in models_losses.values():
            for loss_stats in model_losses.values():
                all_means.extend(loss_stats["mean"])
        
        if len(all_means) > 0:
            data_geometric_mean = np.exp(np.mean(np.log(np.array(all_means)[np.array(all_means) > 0])))
            # Scale the reference to be visible: position it near the geometric mean at mid-range x
            x_mid = x_ref[len(x_ref) // 2]
            C = data_geometric_mean * (x_mid ** 0.5)
        else:
            # Fallback if no data
            C = 1.0
        
        y_ref = C * x_ref ** (-0.5)
        
        (convergence_ref_line,) = ax.plot(
            x_ref,
            y_ref,
            color="red",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=r"$x^{-0.5}$ reference",
        )
        # Create handle for legend
        convergence_ref_handle = plt.Line2D(
            [0], [0], color="red", linestyle=":", linewidth=2, alpha=0.7
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    if y_scale == "log":
        ax.set_yscale("log")
        ax.set_xscale("log")
    # change y scale min
    if bottom_ylim:
        ax.set_ylim(bottom=bottom_ylim)

    # Add per-model training time annotations just above the highest dot per model at each x
    if time_annotations:
        for model_name, times in time_annotations.items():
            if model_name not in model_ymax_per_x:
                continue
            ymax = model_ymax_per_x[model_name]
            n = min(len(times), len(x_arr))
            for j in range(n):
                t_val = times[j]
                if t_val is None or (isinstance(t_val, float) and np.isnan(t_val)):
                    continue
                txt = time_fmt.format(t_val)
                # Place slightly above with an offset in points to be robust to log/linear scales
                ax.annotate(
                    txt,
                    (x_arr[j], ymax[j]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=12,
                    color="black",
                )

    # Format x-axis ticks (scientific) if large
    def sci_fmt(x, pos):  # noqa: ANN001
        if x == 0:
            return "0"
        return f"{x:.1e}".replace("e+0", "e+").replace("e-0", "e-")
    ax.xaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax.set_xticks(x_arr)
    # for label in ax.get_xticklabels():
    #     label.set_rotation(45)
    #     label.set_horizontalalignment("right")

    # Legends: create combined bottom-left positioning without overlap.
    # Place loss legend slightly above model legend.
    loss_legend = ax.legend(
        loss_handles.values(),
        loss_handles.keys(),
        title="Loss type",
        loc=loss_legend_loc,
        bbox_to_anchor=loss_legend_bbox,
    )
    ax.add_artist(loss_legend)
    
    model_legend = ax.legend(
        model_handles.values(),
        model_handles.keys(),
        title="Model",
        loc="lower left",
        bbox_to_anchor=(0.20, 0.02),
    )
    
    # Add convergence reference in a separate legend box to the right
    if convergence_ref_handle is not None:
        ax.add_artist(model_legend)
        ax.legend(
            [convergence_ref_handle],
            [r"$x^{-0.5}$"],
            title="Reference",
            loc="lower left",
            bbox_to_anchor=(0.36, 0.02),
        )

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, save_name)
    plt.savefig(out_path, bbox_inches="tight")

    plt.close()
    # Reset style
    plt.style.use("default")
    return out_path

if __name__ == "__main__":

    # #### Model analysis example
    # Example synthetic data for two models, three losses each
    x_vals = [2e6, 20e6, 250e6]

    models_losses_example = {
        ## Fixed epochs (1000)
        # "BAMPNO": {
        #     "Train (rel. L2)": {"mean": [0.0056, 0.0049, 0.00053], "std": [0.0001, 0.0001, 4.71e-5]},
        #     "Test (rel. L2)": {"mean": [0.015, 0.0086, 0.0079], "std": [0.0003, 0.0002, 0.00005]},
        #     "Test (rel. H1)": {"mean": [0.081, 0.037, 0.027], "std": [0.001, 0.0001, 0.0001]},
        # },
        ## Fixed time (the same of FNO for 1000 epochs)
        "BAMPNO": {
            "Train (rel. L2)": {"mean": [0.01, 0.0062, 0.0010], "std": [0.0001, 0.0001, 4.71e-5]},
            "Test (rel. L2)": {"mean": [0.02, 0.0096, 0.0081], "std": [0.0003, 0.0002, 0.00005]},
            "Test (rel. H1)": {"mean": [0.085, 0.047, 0.03], "std": [0.001, 0.0001, 0.0001]},
        },
        "FNO": {
            "Train (rel. L2)": {"mean": [0.0078, 0.0025, 0.0013], "std": [2e-4, 6e-5, 8e-5]},
            "Test (rel. L2)": {"mean": [0.026, 0.025, 0.023], "std": [2e-4, 0.0007, 0.00016]},
            "Test (rel. H1)": {"mean": [0.098, 0.088, 0.083], "std": [0.0015, 0.0019, 0.0012]},
        },
    }

    # Example training times (hours) per model matching x_vals
    # train_times = {
    #     "BAMPNO": [1.3, 3.6, 7],
    #     "FNO": [0.7, 2.1, 5.2],
    # }
    train_times = None

    x_label = "Number of trainable parameters"
    bottom_ylim = None
    show_convergence_reference = False

    #### Data analysis example
    # Example synthetic data for two models, three losses each
    # x_vals = [300, 600, 1200, 2400]

    # models_losses_example = {
    #     ## Fixed epochs (1000)
    #     # "BAMPNO": {
    #     #     "Train (rel. L2)": {"mean": [0.0074, 0.0051, 0.0046, 0.0045], "std": [0.0001, 0.0001, 4.71e-5, 3e-5]},
    #     #     "Test (rel. L2)": {"mean": [0.024, 0.017, 0.012, 0.007], "std": [0.0003, 0.0002, 0.00005, 0.0001]},
    #     #     "Test (rel. H1)": {"mean": [0.066, 0.054, 0.042, 0.017], "std": [0.001, 0.0001, 0.0001, 0.0002]},
    #     # },
    #     ## Fixed time (the same of FNO for 1000 epochs)
    #     "BAMPNO": {
    #         "Train (rel. L2)": {"mean": [0.0080, 0.0058, 0.0053, 0.0048], "std": [0.0001, 0.0001, 4.71e-5, 3e-5]},
    #         "Test (rel. L2)": {"mean": [0.028, 0.020, 0.013, 0.009], "std": [0.0003, 0.0002, 0.00005, 0.0001]},
    #         "Test (rel. H1)": {"mean": [0.063, 0.06, 0.046, 0.036], "std": [0.001, 0.0001, 0.0001, 0.0002]},
    #     },
    #     "FNO": {
    #         "Train (rel. L2)": {"mean": [0.0045, 0.0034, 0.0031, 0.0026], "std": [2e-4, 6e-5, 8e-5]},
    #         "Test (rel. L2)": {"mean": [0.049, 0.032, 0.021, 0.016], "std": [2e-4, 0.0007, 0.00016]},
    #         "Test (rel. H1)": {"mean": [0.144, 0.1033, 0.081, 0.065], "std": [0.0015, 0.0019, 0.0012]},
    #     },
    # }

    # # Example training times (hours) per model matching x_vals
    # # train_times = {
    # #     "BAMPNO": [1.0, 1.8, 3.6, 7.2],
    # #     "FNO": [0.6, 1.2, 2.1, 4.2],
    # # }
    # train_times = None
    # bottom_ylim = 1e-3
    # x_label = "Number of training samples"
    # show_convergence_reference = True

    #### Data regularity example
    # x_vals = [1, 2, 8]

    # models_losses_example = {
    #     "BAMPNO": {
    #         "Train (rel. L2)": {"mean": [0.0049, 0.0012, 0.1], "std": [0.0001, 0.0001, 4.71e-5]},
    #         "Test (rel. L2)": {"mean": [0.0086, 0.0023, 0.1], "std": [0.0003, 0.0002, 0.00005]},
    #         "Test (rel. H1)": {"mean": [0.037, 0.0070, 0.1], "std": [0.001, 0.0001, 0.0001]},
    #     },
    #     "FNO": {
    #         "Train (rel. L2)": {"mean": [0.0025, 0.0009, 0.0011], "std": [2e-4, 6e-5, 8e-5]},
    #         "Test (rel. L2)": {"mean": [0.025, 0.0045, 0.0045], "std": [2e-4, 0.0007, 0.00016]},
    #         "Test (rel. H1)": {"mean": [0.088, 0.0129, 0.0172], "std": [0.0015, 0.0019, 0.0012]},
    #     },
    # }
    # train_times = None
    # bottom_ylim = 1e-4
    # x_label = "Input regularity"
    # show_convergence_reference = False

    out_path = plot_models_losses(
        x_values=x_vals,
        models_losses=models_losses_example,
        title="Model loss comparison",
        y_label="Relative error",
        x_label=x_label,
        annotate=False,
        time_annotations=train_times,
        time_fmt="{:.1f}h",
        bottom_ylim=bottom_ylim,
        show_convergence_reference=show_convergence_reference,
    )
    print(f"Done. Figure saved to {out_path}")
