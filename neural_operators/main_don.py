import torch
from FNO.FNO_benchmarks import Darcy
from Loss_fun import LprelLoss
from ray import tune
from tune import tune_hyperparameters

from neural_operators.DeepONet.CNNModules import ConvBranch2D
from neural_operators.DeepONet.DeepONet import DeepOnetNoBiasOrg, FeedForwardNN


def main():
    device = torch.device("cpu")
    config_space = {
        "FourierF": tune.choice([0]),
        "N_layers": tune.choice([2, 4, 6, 8]),
        "N_res": tune.randint(0, 5),
        "basis": tune.choice([50, 100, 200, 500]),
        "batch_size": tune.choice([32]),
        "kernel_size": tune.choice([3]),
        "layers": tune.randint(1, 9),
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "neurons": tune.choice([128, 256, 512]),
        "retrain": tune.choice([4]),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "scheduler_step": tune.choice([10]),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
    }

    def model_builder(config):
        branch = ConvBranch2D(
            in_channels=1,
            N_layers=config["N_layers"],
            N_res=config["N_res"],
            kernel_size=config["kernel_size"],
            multiply=32,
            out_channel=config["basis"],
        ).to(device)

        trunk = FeedForwardNN(
            2 * config["FourierF"],
            config["basis"],
            layers=config["layers"],
            neurons=config["neurons"],
            retrain=config["retrain"],
        ).to(device)

        return DeepOnetNoBiasOrg(branch, trunk).to(device)

    dataset_builder = lambda config: Darcy(
        {
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        device,
        config["batch_size"],
        search_path="/",
    )
    loss_fn = LprelLoss(2, False)
    tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        runs_per_cpu=8,
        runs_per_gpu=0,
    )


if __name__ == "__main__":
    main()
