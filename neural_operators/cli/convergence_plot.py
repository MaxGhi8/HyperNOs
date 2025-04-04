"""
File for making the convergence plot respect to some quantities,
for example the number of trainable parameters of the network or the number of samples in the dataset.
The plot is made with the seaborn library and saved in the figures folder.
The values have to be computed from hand and passed to the function.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_conv_plot(iterations: np.array, error_values: np.array, std_devs: np.array):

    df = pd.DataFrame({"Iteration": iterations, "Error": error_values})
    df2 = pd.DataFrame(
        {
            "x": np.array([0.5 * 10**6, 250 * 10**6]),
            # "y": np.array([0.00256, 0.00078]),
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
    plt.tight_layout()
    # plt.show()
    plt.savefig("figures/convergence_plot.png")

    # Resets the style to default
    plt.style.use("default")


if __name__ == "__main__":

    # Define the data (cont_tran)
    iterations = np.array(
        [0.5 * 10**6, 8 * 10**6, 50 * 10**6, 150 * 10**6, 250 * 10**6]
    )
    error_values = np.array([0.0029, 0.0017, 0.00078, 0.00052, 0.00036])
    std_devs = np.array([0.00013, 0.00006, 0.000067, 0.000027, 0.000004])
    plot_conv_plot(iterations, error_values, std_devs)
