"""
File for making the convergence plot respect to some quantities,
for example the number of trainable parameters of the network or the number of samples in the dataset.
The plot is made with the seaborn library and saved in the figures folder.
The values have to be computed from hand and passed to the function.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


def plot_conv_plot(
    iterations: np.array,
    error_values: np.array,
    std_devs: np.array,
    n_y_ticks: int = 8,
):

    df = pd.DataFrame({"Iteration": iterations, "Error": error_values})
    df2 = pd.DataFrame(
        {
            "x": np.array([0.5 * 10**6, 250 * 10**6]),
            # "y": np.array([0.00256, 0.00078]),
            # "y": np.array([(0.5 * 10**6)**(-0.1674), (250 * 10**6)**(-0.1674)]),
            "y": np.array([0.0015, 0.00053]),
        }
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        x="Iteration",
        y="Error",
        data=df,
        marker="o",
        markersize=8,
        color="#1f77b4",
        label="Experimental error bar",
    )

    # Add error bars
    for i in range(len(iterations)):
        x = iterations[i]
        y = error_values[i]
        std = std_devs[i]

        # Add vertical error line
        plt.plot(
            [x, x],
            [y - std, y + std],
            color="#1f77b4",
            linewidth=1.5,
            alpha=0.8,
        )

        # Add horizontal caps at the ends
        cap_width = 500000
        plt.plot(
            [x - cap_width, x + cap_width],
            [y - std, y - std],
            color="#1f77b4",
            linewidth=1.5,
            alpha=0.8,
        )
        plt.plot(
            [x - cap_width, x + cap_width],
            [y + std, y + std],
            color="#1f77b4",
            linewidth=1.5,
            alpha=0.8,
        )

    # Add a trend line
    sns.lineplot(
        x="x",
        y="y",
        data=df2,
        color="red",
        alpha=0.7,
        lw=2,
        ls="--",
        label="Convergence rate",
    )
    # sns.regplot(
    #     x="Iteration",
    #     y="Error",
    #     data=df,
    #     scatter=False,
    #     ci=None,
    #     line_kws={"color": "red", "alpha": 0.7, "lw": 2, "ls": "--"},
    # )

    # Add annotations for each point (slightly offset to not overlap with error bars)
    for i, txt in enumerate(error_values):
        plt.annotate(
            f"{txt:.4f}",
            (iterations[i], error_values[i]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )

    plt.title("Convergence Plot", fontsize=16)
    plt.xlabel("Number of trainable parameters", fontsize=14)
    plt.ylabel("Relative error", fontsize=14)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Add more labels on y-axis (custom log-spaced ticks)
    y_min = error_values.min()
    y_max = error_values.max()
    # Expand slightly for visual padding
    y_min *= 0.8
    y_max *= 1.25
    if y_min <= 0:
        y_min = error_values[error_values > 0].min() * 0.8
    ticks = np.logspace(np.log10(y_min), np.log10(y_max), num=n_y_ticks)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))

    # Format x-axis ticks in scientific notation (mantissa e exponent for each tick)
    ax.set_xticks(iterations)
    def sci_fmt(x, pos):
        if x == 0:
            return "0"
        return f"{x:.1e}".replace("e+0", "e+").replace("e-0", "e-")
    ax.xaxis.set_major_formatter(FuncFormatter(sci_fmt))
    # Rotate x tick labels for readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    
    plt.tight_layout()
    # Ensure output directory exists and save
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "convergence_plot.png"))

    # Resets the style to default
    plt.style.use("default")


def compute_convergence_rate(iterations: np.ndarray, error_values: np.ndarray):
    """Compute global and pairwise convergence rates.

    Returns a dict with keys:
      global_p: least-squares exponent p in Error ~ C * N^{-p}
      C: prefactor
      pairwise_p: list of pairwise exponents between successive points
    """
    logN = np.log(iterations)
    logE = np.log(error_values)
    slope, intercept = np.polyfit(logN, logE, 1)  # logE = slope*logN + intercept
    p_global = -slope
    C = np.exp(intercept)
    pairwise_p = []
    for i in range(len(iterations) - 1):
        p = -(
            (np.log(error_values[i + 1]) - np.log(error_values[i]))
            / (np.log(iterations[i + 1]) - np.log(iterations[i]))
        )
        pairwise_p.append(p)
    return {"global_p": p_global, "C": C, "pairwise_p": pairwise_p}


def compute_reference_rate(x_points: np.ndarray, y_points: np.ndarray):
    """Compute exponent of reference (red) line defined by two points."""
    if len(x_points) != 2 or len(y_points) != 2:
        return None
    p = -(
        (np.log(y_points[1]) - np.log(y_points[0]))
        / (np.log(x_points[1]) - np.log(x_points[0]))
    )
    return p


if __name__ == "__main__":
    # Data
    iterations = np.array([0.5 * 10**6, 8 * 10**6, 50 * 10**6, 150 * 10**6, 250 * 10**6])
    error_values = np.array([0.0029, 0.0017, 0.00078, 0.00052, 0.00036])
    std_devs = np.array([0.00013, 0.00006, 0.000067, 0.000027, 0.000004])

    # Plot
    plot_conv_plot(iterations, error_values, std_devs)

    # Convergence rates
    rates = compute_convergence_rate(iterations, error_values)
    ref_p = compute_reference_rate(
        np.array([0.5 * 10**6, 250 * 10**6]), np.array([0.0015, 0.00053])
    )

    print("Convergence rate analysis (Error ~ C * N^{-p}):")
    print(f"  Global fitted p: {rates['global_p']:.4f}")
    print(f"  Prefactor C: {rates['C']:.4e}")
    print(
        "  Pairwise p values: "
        + ", ".join(f"{p:.3f}" for p in rates["pairwise_p"])
    )
    if ref_p is not None:
        print(f"  Reference line p (red dashed): {ref_p:.4f}")
    print("Done. Figure saved to figures/convergence_plot.png")
    print("Done. Figure saved to figures/convergence_plot.png")
