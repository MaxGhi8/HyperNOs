import torch
import os
from datetime import datetime
import json
import re


def save_best_hypeparameter(results, path, problem):
    save_path = path + "/results/" + problem + "/"

    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    name_file = (
        save_path
        + "best_hyperparameters_"
        + problem
        + "_"
        + str(datetime.now().strftime("%d-%m-%Y-%H:%M"))
        + ".json"
    )
    print(
        f"Best hyperparameters found were: {results.get_best_result(metric='loss_validation', mode='min').config} | Loss validation: {results.get_best_result(metric='loss_validation', mode='min').metrics["loss_validation"]}"
    )

    best_hyperparameters = json.dump(
        results.get_best_result(metric="loss_validation", mode="min").config,
        open(name_file, "w"),
    )


def compute_num_conv_layers(n_input, kernel_size=3, stride=2):
    """
    Compute how many conv layers can be applied before the feature map
    becomes smaller than the kernel size (valid convs).
    """
    n = n_input
    num_layers = 0
    while n >= kernel_size:
        n = (n - kernel_size) // stride + 1
        if n < kernel_size:
            break
        num_layers += 1
    return num_layers, n


class UnitGaussianNormalizer(object):
    """
    Initial normalization is the point-wise gaussian normalization over the tensor x
    dimension: (n_samples)*(nx)*(ny)
    """

    def __init__(self, x, eps=1e-5):
        self.mean = torch.mean(x, 0).to(x.device)
        self.std = torch.std(x, 0).to(x.device)
        self.eps = torch.tensor(eps).to(x.device)

    def encode(self, x):
        x = (x - self.mean.to(x.device)) / (
            self.std.to(x.device) + self.eps.to(x.device)
        )
        return x

    def decode(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        eps = self.eps.to(x.device)
        x = x * (std + eps) + mean
        return x


class minmaxGlobalNormalizer(object):
    """
    Initial normalization is the global min-max normalization over the tensor x
    """

    def __init__(self, x):
        self.min = x
        self.max = x
        for _ in range(x.dim()):
            self.min = torch.min(self.min, dim=0).values
            self.max = torch.max(self.max, dim=0).values

    def encode(self, x):
        min_ = self.min.to(x.device)
        max_ = self.max.to(x.device)
        x = (x - min_) / (max_ - min_)
        return x

    def decode(self, x):
        min_ = self.min.to(x.device)
        max_ = self.max.to(x.device)
        x = x * (max_ - min_) + min_
        return x


def load_latest_hyperparameters(list_json):
    """Load the latest hyperparameters JSON file based on the date in its filename."""
    if not list_json:
        print("No JSON files found in the specified path.")
        return None

    date_pattern = re.compile(r"(\d{1,2}-\d{1,2}-\d{4}-\d{1,2}:\d{2})")

    def extract_datetime(filename):
        match = date_pattern.search(filename)
        if match:
            # Parse date in format: day-month-year-hour:minute
            return datetime.strptime(match.group(1), "%d-%m-%Y-%H:%M")
        return datetime.min  # fallback if no date found

    # Find file with the latest datetime in its name
    latest_json = max(list_json, key=extract_datetime)

    with open(latest_json, "r") as f:
        hyperparameters = json.load(f)

    print(f"Loaded hyperparameters from: {os.path.basename(latest_json)}")
    return hyperparameters
