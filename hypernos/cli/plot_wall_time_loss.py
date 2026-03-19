"""
This script parses CSV files containing wall time and loss values,
and generates a plot of the loss values over time for the ray hyperparameters optimization process.
We need to save the errors in CSVs (one CSV for each trial done in the ray hyperparameters optimization process).
"""

import glob
import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_csv_files(directory_path: str, pattern: str = "*.csv"):

    all_files = glob.glob(os.path.join(directory_path, pattern))

    if not all_files:
        raise ValueError(
            f"No files found in {directory_path} matching pattern {pattern}"
        )

    times = []
    values = []
    steps = []
    for filename in all_files:
        df = pd.read_csv(filename)

        times.append(np.array(df["Wall time"]))
        values.append(np.array(df["Value"]))
        steps.append(np.array(df["Step"]))

    return times, values, steps


def create_wall_time_plot(
    times: list,
    values: list,
    sample_rate: int = 1,
):
    sns.set(style="white", palette="deep")
    fig = plt.figure(figsize=(14, 6))
    plt.rcParams["text.usetex"] = True

    for time, value in zip(times, values):
        # Convert time
        time = np.array([datetime.fromtimestamp(t) for t in time])

        plt.plot(time[10::sample_rate], value[10::sample_rate])

    fig.autofmt_xdate()  # Auto-rotate date labels for better readability
    plt.xlabel("Wall Time (mm-dd hh)", fontsize=14)

    plt.ylabel("$L^1$ rel. loss", fontsize=14)
    plt.yscale("log")
    plt.ylim(3 * 1e-4, 2 * 1e-1)
    plt.grid(True, which="both", ls="-", alpha=0.1, color="black")
    plt.tight_layout()

    return plt.gcf()


# Example usage
if __name__ == "__main__":

    data_directory = "train_loss/loss_cont_tran"

    times, values, steps = parse_csv_files(data_directory)
    fig = create_wall_time_plot(
        times=times,
        values=values,
        sample_rate=10,
    )

    fig.savefig("wall_time_plot.png", dpi=300, bbox_inches="tight")
    # plt.show()
