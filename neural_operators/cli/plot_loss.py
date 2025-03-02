import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_loss_curves(
    train_files_list: list[str],
    test_files_list: list[str],
    test_L1: list[str],
    test_H1: list[str],
    path: str = None,
    y_min: float = None,
    y_max: float = None,
):
    """
    Create a plot of training and test loss curves from multiple CSV files,
    showing mean and standard deviation.

    Parameters:
    train_files_list (list): List of paths to training CSV files
    test_files_list (list): List of paths to test CSV files
    test_L1 (list): List of paths with the L^1 rel. test loss
    test_H1 (list): List of paths with the H^1 rel. test loss
    path (str): Path to save the plot
    y_min (float): Minimum value for the y-axis
    y_max (float): Maximum value for the y-axis
    """
    train_dfs = []
    test_dfs = []
    L1_dfs = []
    H1_dfs = []

    for train_file in train_files_list:
        df = pd.read_csv(train_file)
        train_dfs.append(df)

    for test_file in test_files_list:
        df = pd.read_csv(test_file)
        test_dfs.append(df)

    for test_file in test_L1:
        df = pd.read_csv(test_file)
        L1_dfs.append(df)

    for test_file in test_H1:
        df = pd.read_csv(test_file)
        H1_dfs.append(df)

    # Get common x-axis values (epochs)
    x = train_dfs[0][train_dfs[0].columns[1]].values

    # Extract y values for all files
    y_train_values = np.array([df[df.columns[2]].values for df in train_dfs])
    y_test_values = np.array([df[df.columns[2]].values for df in test_dfs])
    y_L1_values = np.array([df[df.columns[2]].values for df in L1_dfs])
    y_H1_values = np.array([df[df.columns[2]].values for df in H1_dfs])

    # Calculate mean and std
    train_mean = np.mean(y_train_values, axis=0)
    train_std = np.std(y_train_values, axis=0)
    test_mean = np.mean(y_test_values, axis=0)
    test_std = np.std(y_test_values, axis=0)
    L1_mean = np.mean(y_L1_values, axis=0)
    L1_std = np.std(y_L1_values, axis=0)
    H1_mean = np.mean(y_H1_values, axis=0)
    H1_std = np.std(y_H1_values, axis=0)

    # Plotting
    sns.set(style="white", palette="deep")
    plt.figure(figsize=(8, 6))
    plt.rcParams["text.usetex"] = True

    plt.plot(x, L1_mean, label=r"Test loss (relative $L^1$)", color="#2ca02c")
    plt.fill_between(
        x,
        L1_mean - L1_std,
        L1_mean + L1_std,
        alpha=0.2,
        color="#2ca02c",
        # label="Test (±std)",
    )

    plt.plot(x, H1_mean, label=r"Test loss (relative $H^1$)", color="#d62728")
    plt.fill_between(
        x,
        H1_mean - H1_std,
        H1_mean + H1_std,
        alpha=0.2,
        color="#d62728",
        # label="Test (±std)",
    )

    plt.plot(x, train_mean, label=r"Train loss (relative $L^2$)", color="#1f77b4")
    plt.fill_between(
        x,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="#1f77b4",
        # label="Train (±std)",
    )

    plt.plot(x, test_mean, label=r"Test loss (relative $L^2$)", color="#ff7f0e")
    plt.fill_between(
        x,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2,
        color="#ff7f0e",
        # label="Test (±std)",
    )

    plt.yscale("log")
    plt.xlabel(r"Epochs")
    plt.ylabel(r"Value of the loss function")
    plt.grid(True, which="both", ls="-", alpha=0.1, color="black")
    plt.legend()
    if y_min and y_max:
        plt.ylim(y_min, y_max)

    plt.tight_layout()

    if path:
        plt.savefig(path)
    else:
        plt.show()


if __name__ == "__main__":

    example = "ord"  # select the example
    mode = "best"  # select the mode of the test
    example2name = {"fhn": "FitzHughNagumo", "hh": "HodgkinHuxley", "ord": "OHaraRudy"}
    example2min = {"fhn": 2e-3, "hh": 8e-3, "ord": 5e-2}
    example2max = {"fhn": 1.5, "hh": 1.5, "ord": 1.5}

    train_files = [
        f"train_loss/{example}_{mode}/0/FNO_1D_{example2name[example]}_Train loss.csv",
        f"train_loss/{example}_{mode}/1/FNO_1D_{example2name[example]}_Train loss.csv",
        f"train_loss/{example}_{mode}/2/FNO_1D_{example2name[example]}_Train loss.csv",
        f"train_loss/{example}_{mode}/3/FNO_1D_{example2name[example]}_Train loss.csv",
        f"train_loss/{example}_{mode}/4/FNO_1D_{example2name[example]}_Train loss.csv",
    ]
    test_files = [
        f"train_loss/{example}_{mode}/0/FNO_1D_{example2name[example]}_Test rel. L^2 error.csv",
        f"train_loss/{example}_{mode}/1/FNO_1D_{example2name[example]}_Test rel. L^2 error.csv",
        f"train_loss/{example}_{mode}/2/FNO_1D_{example2name[example]}_Test rel. L^2 error.csv",
        f"train_loss/{example}_{mode}/3/FNO_1D_{example2name[example]}_Test rel. L^2 error.csv",
        f"train_loss/{example}_{mode}/4/FNO_1D_{example2name[example]}_Test rel. L^2 error.csv",
    ]
    test_L1 = [
        f"train_loss/{example}_{mode}/0/FNO_1D_{example2name[example]}_Test rel. L^1 error.csv",
        f"train_loss/{example}_{mode}/1/FNO_1D_{example2name[example]}_Test rel. L^1 error.csv",
        f"train_loss/{example}_{mode}/2/FNO_1D_{example2name[example]}_Test rel. L^1 error.csv",
        f"train_loss/{example}_{mode}/3/FNO_1D_{example2name[example]}_Test rel. L^1 error.csv",
        f"train_loss/{example}_{mode}/4/FNO_1D_{example2name[example]}_Test rel. L^1 error.csv",
    ]
    test_H1 = [
        f"train_loss/{example}_{mode}/0/FNO_1D_{example2name[example]}_Test rel. H^1 error.csv",
        f"train_loss/{example}_{mode}/1/FNO_1D_{example2name[example]}_Test rel. H^1 error.csv",
        f"train_loss/{example}_{mode}/2/FNO_1D_{example2name[example]}_Test rel. H^1 error.csv",
        f"train_loss/{example}_{mode}/3/FNO_1D_{example2name[example]}_Test rel. H^1 error.csv",
        f"train_loss/{example}_{mode}/4/FNO_1D_{example2name[example]}_Test rel. H^1 error.csv",
    ]

    plot_loss_curves(
        train_files,
        test_files,
        test_L1,
        test_H1,
        f"train_loss/loss_function_{example.upper()}_{mode}.png",
        example2min[example],
        example2max[example],
    )
