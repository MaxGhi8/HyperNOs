import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_loss_curves(train_files_list: list[str], test_files_list: list[str]):
    """
    Create a plot of training and test loss curves from multiple CSV files,
    showing mean and standard deviation.

    Parameters:
    train_files_list (list): List of paths to training CSV files
    test_files_list (list): List of paths to test CSV files
    """
    train_dfs = []
    test_dfs = []

    for train_file in train_files_list:
        df = pd.read_csv(train_file)
        train_dfs.append(df)

    for test_file in test_files_list:
        df = pd.read_csv(test_file)
        test_dfs.append(df)

    # Get common x-axis values (epochs)
    x_train = train_dfs[0][train_dfs[0].columns[1]].values
    x_test = test_dfs[0][test_dfs[0].columns[1]].values

    # Extract y values for all files
    y_train_values = np.array([df[df.columns[2]].values for df in train_dfs])
    y_test_values = np.array([df[df.columns[2]].values for df in test_dfs])

    # Calculate mean and std
    train_mean = np.mean(y_train_values, axis=0)
    train_std = np.std(y_train_values, axis=0)
    test_mean = np.mean(y_test_values, axis=0)
    test_std = np.std(y_test_values, axis=0)

    # Plotting
    sns.set(style="white", palette="deep")
    plt.figure(figsize=(10, 6))

    plt.plot(x_train, train_mean, label="Train (mean)", color="#1f77b4")
    plt.fill_between(
        x_train,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="#1f77b4",
        label="Train (±std)",
    )

    plt.plot(x_test, test_mean, label="Test (mean)", color="#ff7f0e")
    plt.fill_between(
        x_test,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2,
        color="#ff7f0e",
        label="Test (±std)",
    )

    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Value of the loss function")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
