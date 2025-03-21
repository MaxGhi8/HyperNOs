import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat


#########################################
# Overlapped Histograms Plot routines
#########################################
def plot_overlapped_histograms(
    errors: list,
    str_norm: str,
    labels: list[str] = None,
):

    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid", palette="deep")
    plt.figure(figsize=(10, 6))

    if labels is None:
        labels = [f"Error {i+1}" for i in range(len(errors))]

    for error, label in zip(errors, labels):
        sns.histplot(
            error,
            bins=100,
            kde=True,
            edgecolor="black",
            label=label,
            alpha=0.6,
        )

    plt.xlabel("Relative Error", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xscale("log")
    plt.title("Histogram of the Relative Error", fontsize=14, pad=20)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.savefig(f"figures/histogram_{str_norm}.png")

    # Resets the style to default
    plt.style.use("default")


#########################################
# Overlapped Swarm-plot routines
#########################################
def plot_errors_swarmplot(
    errors: list,
    str_norm: str,
    labels: list[str] = None,
):
    sns.set(style="whitegrid", palette="deep")
    plt.figure(figsize=(10, 6))

    if labels is None:
        labels = [f"Error {i+1}" for i in range(len(errors))]

    data = []
    for error, label in zip(errors, labels):
        for err_val in error:
            data.append(
                {"Hyperparameters configuration": label, "Relative Error": err_val}
            )

    df = pd.DataFrame(data)

    ax = sns.swarmplot(
        x="Hyperparameters configuration",
        y="Relative Error",
        hue="Hyperparameters configuration",  # Assign hue instead of just palette
        data=df,
        size=3,
        alpha=0.7,
        legend=False,  # No need for legend as it's redundant with x-axis labels
    )

    # Add a boxplot to show summary statistics
    # sns.boxplot(
    #     x="Hyperparameters configuration",
    #     y="Relative Error",
    #     data=df,
    #     width=0.5,
    #     boxprops=dict(alpha=0.3),
    #     showfliers=False,
    # )

    plt.ylabel("Relative Error", fontsize=12)
    plt.yscale("log")
    plt.title("Swarmplot of the Relative Error", fontsize=14, pad=20)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.savefig(f"figures/swarmplot_{str_norm}.png")

    # Resets the style to default
    plt.style.use("default")


if __name__ == "__main__":
    folder = "../../data/mishra/outputs_for_website/"

    problem = "cont_tran"
    model = "FNO"
    hyperparams_modes = ["default", "best"]
    labels = ["sota hyperparams", "our best hyperparams"]

    errors = []
    for hyperparams_mode in hyperparams_modes:
        # Load the data
        data = loadmat(f"{folder}/{problem}_{model}_trainL1_{hyperparams_mode}.mat")

        # Compute the relative error
        num_examples = data["output"].shape[0]
        diff_norms = np.linalg.norm(
            data["output"].reshape(num_examples, -1)
            - data["prediction"].reshape(num_examples, -1),
            ord=1,
            axis=1,
        )
        y_norms = np.linalg.norm(
            data["output"].reshape(num_examples, -1), ord=1, axis=1
        )
        if np.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")
        relative_diff = diff_norms / y_norms

        errors.append(relative_diff)

    # Plot the overlapped histograms
    plot_overlapped_histograms(errors, "L1", labels)

    # Plot the swarm-plot
    plot_errors_swarmplot(errors, "L1", labels)
