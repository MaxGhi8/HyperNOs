import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curves(csv_file_train, csv_file_test):
    """
    Create a plot of training and test loss curves from CSV data.

    Parameters:
    """
    # Read the CSV file
    df_train = pd.read_csv(csv_file_train)
    df_test = pd.read_csv(csv_file_test)

    # Create the figure and axis
    plt.figure(figsize=(10, 6))
    plt.plot(df_train["Step"], df_train["Value"], label="Train", color="#1f77b4")
    plt.plot(df_test["Step"], df_test["Value"], label="Test", color="#ff7f0e")

    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Value of the loss function")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    # Tight layout to prevent label clipping
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_loss_curves(
        "FNO_1D_OHaraRudy_Train loss.csv", "FNO_1D_OHaraRudy_Test rel. L^2 error.csv"
    )
