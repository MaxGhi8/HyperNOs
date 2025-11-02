from DON import *
import torch
import torch.optim as optim
from train import *
import os
from ray import tune, init
from utilities import *
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import json
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from utilities import *
import pickle
import ray

current_path = os.getcwd()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)


problem = "fhn_1d"
data_path = current_path + "/data/" + problem + "/"

if problem == "poisson":
    batch_size = 10
    n_points = [50, 50]

    x = torch.tensor(np.linspace(0, np.pi / 2), dtype=torch.float32)
    y = torch.tensor(np.linspace(0, np.pi / 2), dtype=torch.float32)

    x_grid, y_gird = torch.meshgrid(x, y)

    sol = lambda a, b, c: c * torch.sin(a * x_grid) * torch.sin(b * y_gird)
    c_train = 1.5 + 0.5 * torch.rand(n_points[0])
    c_test = 1.5 + 0.5 * torch.rand(n_points[0])

    u_train = torch.empty([50, n_points[0], n_points[1]])
    u_test = torch.empty([50, n_points[0], n_points[1]])

    for index, c_t in enumerate(c_train):
        u_train[index, :, :] = torch.tensor(sol(2, 2, c_t), dtype=torch.float32)

    for index, c_t in enumerate(c_test):
        u_test[index, :, :] = torch.tensor(sol(2, 2, c_t), dtype=torch.float32)

    c_train = c_train[:, None, None] * torch.ones((1, 50, 50), dtype=torch.float32)
    c_test = c_test[:, None, None] * torch.ones((1, 50, 50), dtype=torch.float32)

    train_dataset = DataLoader(
        TensorDataset(c_train.unsqueeze(1), u_train),
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataset = DataLoader(
        TensorDataset(c_test.unsqueeze(1), u_test),
        batch_size=batch_size,
        shuffle=True,
    )

    coords = torch.stack([x_grid.flatten(), y_gird.flatten()], axis=1).to(
        device, dtype=torch.float32
    )

    prob_dim = 2
    n_variables = 1

elif problem == "fhn_1d":
    batch_size = 32
    n_points = [100, 100]
    x = torch.linspace(0, 1, 100)
    t = torch.linspace(0, 40, 100)

    T, X = torch.meshgrid(t, x)
    coords = torch.stack([T.flatten(), X.flatten()], axis=1).to(device)

    file_training = open(data_path + "fhn_1d_training.pkl", "rb")
    dataset_training = pickle.load(file_training)
    file_training.close()
    # transform the input from [n_example] to [n_example,n_pts,n_pts]
    input_training = torch.tensor(dataset_training["input"])

    voltage_training = torch.tensor(dataset_training["Voltage"])
    gating_training = torch.tensor(dataset_training["gating"])

    input_normalizer = minmaxGlobalNormalizer(input_training)
    voltage_normalizer = minmaxGlobalNormalizer(voltage_training)
    gating_normalizer = minmaxGlobalNormalizer(gating_training)

    train_dataset = DataLoader(
        TensorDataset(
            input_normalizer.encode(input_training).unsqueeze(1),
            voltage_normalizer.encode(voltage_training),
            gating_normalizer.encode(gating_training),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # # load the test
    # file_test = open(data_path + "fhn_1d_test.pkl", "rb")
    # dataset_test = pickle.load(file_test)
    # file_test.close()
    # # transform the input from [n_example] to [n_example,n_pts,n_pts]
    # input_test = torch.tensor(dataset_test["input"])

    # voltage_test = torch.tensor(dataset_test["Voltage"])
    # gating_test = torch.tensor(dataset_test["gating"])

    # test_dataset = DataLoader(
    #     TensorDataset(
    #         input_normalizer.encode(input_test).unsqueeze(1),
    #         voltage_normalizer.encode(voltage_test),
    #         gating_normalizer.encode(gating_test),
    #     ),
    #     batch_size=batch_size,
    #     shuffle=True,
    # )

    file_validation = open(data_path + "fhn_1d_validation.pkl", "rb")
    dataset_validation = pickle.load(file_validation)
    file_validation.close()
    # transform the input from [n_example] to [n_example,n_pts,n_pts]
    input_validation = torch.tensor(dataset_validation["input"])

    voltage_validation = torch.tensor(dataset_validation["Voltage"])
    gating_validation = torch.tensor(dataset_validation["gating"])

    validation_dataset = DataLoader(
        TensorDataset(
            input_normalizer.encode(input_validation).unsqueeze(1),
            voltage_normalizer.encode(voltage_validation),
            gating_normalizer.encode(gating_validation),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    prob_dim = 2
    n_variables = 2


def train_DON_ray(config, train_dataset, validation_dataset, coords, print_loss=False):
    torch.set_default_dtype(torch.float32)
    # define the DON
    branch_hyperparameters = config["branch"]
    trunk_hyperparameters = config["trunk"]
    DON_hyperparameter = config["DON"]

    model = DeepONet(branch_hyperparameters, trunk_hyperparameters, DON_hyperparameter)
    model.to(device, dtype=torch.float32)

    if tune.get_checkpoint():
        loaded_checkpoint = tune.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    # define the parameters for the optimization
    hyperparameters_optimizers = config["train"]
    num_epochs = hyperparameters_optimizers["num_epochs"]
    learning_rate = hyperparameters_optimizers["learning_rate"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=hyperparameters_optimizers["weight_decay"],
    )
    gamma = hyperparameters_optimizers["gamma"]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
    norm_train = hyperparameters_optimizers["norm_p"]
    norm_type = hyperparameters_optimizers["norm_type"]
    n_output = DON_hyperparameter["n_output"]

    if n_output == 1:
        loss_values = []
        # loss_test = np.array([[] * len([1, 2]) for _ in range(n_output + 1)])
        loss_test = [[] for _ in [1, 2]]
        loss_train = [[]]
    else:
        loss_values = [[] for _ in range(n_output + 1)]
        # loss_test = np.array([[] * len([1, 2]) for _ in range(n_output + 1)])
        # loss_test = np.zeros((n_output + 1, len([1, 2]), 0))
        loss_train = np.zeros((n_output + 1, len([norm_train]), 0))
        loss_validation = np.zeros((n_output + 1, len([norm_train]), 0))

    # loss_train = np.array([[] * len([2]) for _ in range(n_output + 1)])
    for epoch in range(num_epochs):
        loss_values = train_epoch(
            model,
            train_dataset,
            coords,
            optimizer,
            loss_values,
            norm_train,
            n_output,
            norm_type,
        )

        # loss_train = eval_loss(
        #     model,
        #     train_dataset,
        #     coords,
        #     loss_train,
        #     [norm_train],
        #     n_output,
        #     norm_type,
        # )

        # loss_test = eval_loss(
        #     model,
        #     test_dataset,
        #     coords,
        #     loss_test,
        #     [1, 2],
        #     n_output,
        #     norm_type,
        # )

        loss_validation = eval_loss(
            model,
            validation_dataset,
            coords,
            loss_validation,
            [norm_train],
            n_output,
            norm_type,
        )
        scheduler.step()
        if problem == "poisson":
            metrics = {"loss_validation": loss_validation[1][-1]}
        elif problem == "fhn_1d":
            metrics = {"loss_validation": loss_validation[0][0][-1]}
        tune.report(metrics)

        if print_loss:
            if n_output == 1:
                if (epoch + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Loss train: {loss_train[0][-1]:.4e}"
                    )
            elif problem == "fhn_1d":
                if (epoch + 1) % 1 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Loss Validation: {loss_validation[0, 0, -1]:.4e},  Loss V: {loss_validation[1, 0, -1]:.4e},  Loss w: {loss_validation[2, 0, -1]:.4e}"
                    )


def tune_DON():

    init(
        address="auto",
        runtime_env={
            "env_vars": {"PYTHONPATH": os.path.abspath("..")},
        },
    )
    config_DON = {
        "n_output": n_variables,
        "n_basis": tune.sample_from(
            lambda spec: np.random.randint(
                (40 + spec.config["DON"]["n_output"] - 1)
                // spec.config["DON"]["n_output"],
                400 // spec.config["DON"]["n_output"] + 1,
            )
            * spec.config["DON"]["n_output"]
        ),
        "dim": prob_dim,
    }

    config_branch = {
        "n_inputs": 1,
        "n_points": n_points,
        "stride": 2,
        "kernel_size": tune.randint(2, 7),
        "channels_conv": tune.sample_from(
            lambda spec: (
                lambda n_layers: (
                    np.cumsum(np.random.randint(1, 30, size=n_layers))
                    + np.random.randint(20, 40)
                ).tolist()
            )(
                compute_num_conv_layers(
                    n_input=spec.config["branch"]["n_points"][0],
                    kernel_size=spec.config["branch"]["kernel_size"],
                    stride=spec.config["branch"]["stride"],
                )[0]
            )
        ),
        "output_dim_conv": tune.sample_from(
            lambda spec: (
                compute_num_conv_layers(
                    n_input=spec.config["branch"]["n_points"][0],
                    kernel_size=spec.config["branch"]["kernel_size"],
                    stride=spec.config["branch"]["stride"],
                )[1]
            )
        ),
        "hidden_layer": tune.sample_from(
            lambda spec: np.random.randint(
                50, 300, size=np.random.randint(1, 5)
            ).tolist()
        ),
        "residual": tune.choice([True, False]),
        "act_fun": tune.choice(["relu", "gelu", "tanh", "leaky_relu", "silu"]),
    }

    config_trunk = {
        "n_inputs": 2,
        "hidden_layer": tune.sample_from(
            lambda spec: np.random.randint(
                50, 300, size=np.random.randint(1, 5)
            ).tolist()
        ),
        "residual": tune.choice([True, False]),
        "act_fun": tune.choice(["relu", "gelu", "tanh", "leaky_relu", "silu"]),
    }

    config_train = {
        "num_epochs": 1000,
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "gamma": tune.quniform(0.75, 0.99, 0.01),
        "norm_p": 2,
        "batch_size": batch_size,
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "norm_type": True,
    }

    config_space = {
        "DON": config_DON,
        "branch": config_branch,
        "trunk": config_trunk,
        "train": config_train,
    }
    scheduler = ASHAScheduler(
        metric="loss_validation",
        mode="min",
        time_attr="training_iteration",
        max_t=1000,
        grace_period=500,
        reduction_factor=2,
        stop_last_trials=True,
    )

    search_alg = HyperOptSearch(
        metric="loss_validation",
        mode="min",
        n_initial_points=20,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_DON_ray,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                coords=coords,
                print_loss=False,
            ),
            resources={"cpu": 1, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=100,
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
        ),
        param_space=config_space,
    )

    results = tuner.fit()
    return results


results = tune_DON()

save_best_hypeparameter(results, current_path, problem)
