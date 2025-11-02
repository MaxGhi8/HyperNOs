import torch
import torch.nn as nn
import numpy as np
from DON import DeepONet
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)


def Lp_rel(u, u_theta, p):
    batch_size = u.shape[0]
    diff = u_theta.reshape(batch_size, -1) - u.reshape(batch_size, -1)
    norm_diff = torch.norm(diff, p=p, dim=1)
    norm_u = torch.norm(u.reshape(batch_size, -1), p=p, dim=1)
    return torch.sum(norm_diff / norm_u)


def Lp(u, u_theta, p):
    print("non questa norma!")
    # TODO da sistemare per il caso 2d
    batch_size = u.shape[0]
    diff = u_theta.reshape(batch_size, -1) - u.reshape(batch_size, -1)
    norm_diff = torch.norm(diff, p=p, dim=1) / (u.shape[1]) ** (1.0 / p)

    return torch.sum(norm_diff)


def train_epoch(
    model, dataset, x, optimizer, loss_values, norm_p, n_output=1, rel=True
):
    model.train()
    n_batches = 0
    epoch_loss = 0.0
    epoch_loss_single = [0.0] * n_output

    for a, *u in dataset:
        a = a.to(device)
        u = [tensor.to(device) for tensor in u]
        optimizer.zero_grad()
        model_output = model(a, x)
        if n_output != 1:
            loss_single = [0.0] * n_output
            loss = [0.0]
            u = torch.stack(u, dim=-1)  # [batch,points,outputs]
            for i in range(n_output):
                if rel == True:
                    loss_single[i] = Lp_rel(u[..., i], model_output[..., i], norm_p)
                else:
                    loss_single[i] = Lp(u[..., i], model_output[..., i], norm_p)
                epoch_loss_single[i] += loss_single[i].detach().item()
            loss = sum(loss_single) / n_output
        else:
            u = u[0]
            if rel == True:
                loss = Lp_rel(u, model_output, norm_p)
            else:
                loss = Lp(u, model_output, norm_p)
            epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        n_batches += a.shape[0]
    if n_output != 1:
        loss_values[0].append(sum(epoch_loss_single) / (n_batches * n_output))
        for i in range(n_output):
            loss_values[i + 1].append(epoch_loss_single[i] / n_batches)
    else:
        loss_values.append(epoch_loss / n_batches)
    return loss_values


def eval_loss(model, dataset, x, loss_history, p_values, n_output=1, rel=True):
    model.eval()
    n_batches = 0
    if n_output != 1:
        batch_losses_sum = np.array([[0.0] * len(p_values) for _ in range(n_output)])
        for a, *u in dataset:
            a = a.to(device)
            u = [tensor.to(device) for tensor in u]
            model_output = model(a, x)
            u = torch.stack(u, dim=-1)
            loss_single = np.array([[0.0] * len(p_values) for _ in range(n_output)])
            for i, p in enumerate(p_values):
                for j in range(n_output):
                    if rel == True:
                        loss_single[j, i] = Lp_rel(
                            u[..., j], model_output[..., j], p
                        ).detach()
                    else:
                        loss_single[j, i] = Lp(u[:, :, j], model_output[:, :, j], p)
                    batch_losses_sum[j, i] += loss_single[j, i].item()

            n_batches += a.shape[0]
        loss_value = np.zeros((n_output + 1, len(p_values)))
        for i, p in enumerate(p_values):
            loss_value[0, i] = sum(batch_losses_sum[:, i]) / (n_batches * n_output)
            for j in range(n_output):
                epoch_avg_losses = batch_losses_sum[j, i] / n_batches
                loss_value[j + 1, i] = epoch_avg_losses
        loss_history = np.concatenate(
            (loss_history, loss_value[:, :, np.newaxis]), axis=2
        )
    else:
        batch_losses_sum = [0.0] * len(p_values)
        for a, u in dataset:
            a = a.to(device)
            u = u.to(device)
            model_output = model(a, x)
            for i, p in enumerate(p_values):
                if rel == True:
                    loss_rel = Lp_rel(u, model_output, p)
                else:
                    loss_rel = Lp(u, model_output, p)
                batch_losses_sum[i] += loss_rel.item()
            n_batches += a.shape[0]
        for i, p in enumerate(p_values):
            epoch_avg_losses = batch_losses_sum[i] / n_batches
            loss_history[i].append(epoch_avg_losses)
    return loss_history


def train_DON(config, train_dataset, test_dataset, coords, print_loss=False):
    torch.set_default_dtype(torch.float32)
    # define the DON
    branch_hyperparameters = config["branch"]
    trunk_hyperparameters = config["trunk"]
    DON_hyperparameter = config["DON"]

    model = DeepONet(branch_hyperparameters, trunk_hyperparameters, DON_hyperparameter)
    model.to(device, dtype=torch.float32)

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
        loss_test = np.zeros((n_output + 1, len([1, 2]), 0))
        loss_train = np.zeros((n_output + 1, len([norm_train]), 0))
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

        loss_train = eval_loss(
            model,
            train_dataset,
            coords,
            loss_train,
            [norm_train],
            n_output,
            norm_type,
        )

        loss_test = eval_loss(
            model,
            test_dataset,
            coords,
            loss_test,
            [1, 2],
            n_output,
            norm_type,
        )

        scheduler.step()

        if print_loss:
            if n_output == 1:
                if (epoch + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Loss train: {loss_train[0][-1]:.4e}"
                    )
            elif n_output == 2:
                if (epoch + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}]\n"
                        f"Train: Loss: {loss_train[0, 0, -1]:.4e},  Loss V: {loss_train[1, 0, -1]:.4e},  Loss w: {loss_train[2, 0, -1]:.4e},\n"
                        f"Test: Loss: {loss_test[0, 1, -1]:.4e},  Loss V: {loss_test[1, 1, -1]:.4e},  Loss w: {loss_test[2, 1, -1]:.4e}",
                    )
    return model, loss_train, loss_test
