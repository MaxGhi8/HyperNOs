import os
import tempfile

import torch
from Loss_fun import (
    H1relLoss,
    H1relLoss_1D,
    H1relLoss_1D_multiout,
    H1relLoss_multiout,
    LprelLoss,
    LprelLoss_multiout,
)
from ray import train
from ray.train import Checkpoint
from tensorboardX import SummaryWriter
from tqdm import tqdm

from neural_operators.utilities import count_params, plot_data


def train_model(config, model, train_loader, val_loader, loss_fn, max_epochs, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"]
    )

    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state"])

    for ep in range(start_epoch, max_epochs):
        # Train the model for one epoch
        train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)

        # Validate the model for one epoch
        acc = validate_epoch(model, val_loader, loss_fn, device)

        if ep % config["checkpoint_frequency"] == 0 or ep == max_epochs - 1:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report({"relative_loss": acc}, checkpoint=checkpoint)
        else:
            train.report({"relative_loss": acc})


def train_model_without_ray(
    config, model, dataset, loss_fn, max_epochs, device, experiment_name
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"]
    )

    folder = f"./experiments/{experiment_name}"
    name_model = f"./experiments/model_{experiment_name}"

    writer = SummaryWriter(log_dir=folder)  # tensorboard
    start_epoch = 0
    ep_step = 50
    plotting = False

    for epoch in range(start_epoch, max_epochs):
        with tqdm(
            desc=f"Epoch {epoch}", bar_format="{desc}: [{elapsed_s:.2f}{postfix}]"
        ) as tepoch:
            train_epoch_result = train_epoch(
                model,
                dataset.train_loader,
                optimizer,
                scheduler,
                loss_fn,
                device,
                tepoch,
                4,
            )
            if epoch == 0 and train_epoch_result is not None:
                esempio_test, soluzione_test = train_epoch_result

            # test the model for one epoch
            (
                test_relative_l1,
                test_relative_l2,
                test_relative_semih1,
                test_relative_h1,
                train_loss,
            ) = validate_epoch2(
                model,
                dataset.test_loader,
                dataset.train_loader,
                loss_fn,
                device,
                tepoch,
            )

            # save the results of train and test on tensorboard
            writer.add_scalars(
                experiment_name,
                {
                    "Train loss": train_loss,
                    "Test rel. L^1 error": test_relative_l1,
                    "Test rel. L^2 error": test_relative_l2,
                    "Test rel. semi-H^1 error": test_relative_semih1,
                    "Test rel. H^1 error": test_relative_h1,
                },
                epoch,
            )

            # make plots with loss separated for every component of the output
            if config["d_u"] > 1:
                (
                    test_relative_l1_multiout,
                    test_relative_l2_multiout,
                    test_relative_semih1_multiout,
                    test_relative_h1_multiout,
                ) = test_fun_multiout(
                    model,
                    dataset.test_loader,
                    device,
                    config["d_u"],
                )
                for i in range(config["d_u"]):
                    writer.add_scalars(
                        f"{experiment_name}_output_{i}",
                        {
                            "Test rel. L^1 error": test_relative_l1_multiout[i],
                            "Test rel. L^2 error": test_relative_l2_multiout[i],
                            "Test rel. semi-H^1 error": test_relative_semih1_multiout[
                                i
                            ],
                            "Test rel. H^1 error": test_relative_h1_multiout[i],
                        },
                        epoch,
                    )

            total_params, total_bytes = count_params(model)
            total_mb = total_bytes / (1024**2)
            with open(folder + "/errors.txt", "w") as file:
                file.write("Training loss: " + str(train_loss) + "\n")
                file.write("Test relative L^1 error: " + str(test_relative_l1) + "\n")
                file.write("Test relative L^2 error: " + str(test_relative_l2) + "\n")
                file.write(
                    "Test relative semi-H^1 error: " + str(test_relative_semih1) + "\n"
                )
                file.write("Test relative H^1 error: " + str(test_relative_h1) + "\n")
                file.write("Current Epoch: " + str(epoch) + "\n")
                file.write(f"Total Parameters: {total_params:,}\n")
                file.write(
                    f"Total Model Size: {total_bytes:,} bytes ({total_mb:.2f} MB)\n"
                )

            # plot data during the training and save on tensorboard
            if epoch == 0:
                # plot the input data
                plot_data(
                    example,
                    esempio_test,
                    [],
                    "Input function",
                    epoch,
                    writer,
                    dataset_str,
                    problem_dim,
                    plotting,
                )

                # plot the exact solution
                plot_data(
                    example,
                    soluzione_test,
                    [],
                    "Exact solution",
                    epoch,
                    writer,
                    dataset_str,
                    problem_dim,
                    plotting,
                )

            # Approximate solution with NO
            if epoch % ep_step == 0:
                with torch.no_grad():  # no grad for efficiency
                    out_test = model(esempio_test.to(device))
                    out_test = out_test.cpu()

                # plot the approximate solution
                plot_data(
                    example,
                    out_test,
                    [],
                    f"Approximate solution with {arc}",
                    epoch,
                    writer,
                    dataset_str,
                    problem_dim,
                    plotting,
                )

                # Module of the difference
                diff = torch.abs(out_test - soluzione_test)
                plot_data(
                    example,
                    diff,
                    [],
                    "Module of the error",
                    epoch,
                    writer,
                    dataset_str,
                    problem_dim,
                    plotting,
                )

        writer.flush()  # for saving final data
        writer.close()  # close the tensorboard writer

        torch.save(model, name_model)


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    loss,
    device: torch.device,
    tepoch=None,
    n_idx: int = -1,
):
    """
    function to train the model, this function is called at each epoch.

    model: the model to train
    train_loader: the training data loader
    optimizer: the optimizer policy
    scheduler: the scheduler policy
    loss: the loss function to be used during training
    device: the device where we have to store all the things
    tepoch: the tqdm object to print the progress
    n_idx: the number of samples to extract from the first batch for the plot on tensorboard
    """
    model.train()
    train_loss = 0.0
    for step, (input_batch, output_batch) in enumerate(train_loader):
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)

        optimizer.zero_grad()  # annealing the gradient

        # compute the output
        output_pred_batch = model.forward(input_batch)

        # extract the first batch for the plot on tensorboard
        if (step == 0) and (n_idx > 0):
            esempio_test = input_batch[:n_idx].cpu()
            soluzione_test = output_batch[:n_idx].cpu()

        # TODO: adjust loss functions for p=3,4
        loss_f = loss(output_pred_batch, output_batch)

        # back propagation
        loss_f.backward()
        optimizer.step()

        # set the postfix for print
        train_loss += loss_f.item()
        if tepoch is not None:
            tepoch.set_postfix(
                {
                    "Batch": step + 1,
                    "Train loss (in progress)": train_loss
                    / (input_batch.shape[0] * (step + 1)),
                }
            )

    # update the learning rate after an epoch
    scheduler.step()

    if n_idx > 0:
        return esempio_test, soluzione_test
    else:
        return None


def validate_epoch(
    model,
    val_loader,
    loss_fn,
    device: torch.device,
):
    with torch.no_grad():
        model.eval()
        loss = 0.0
        examples_count = 0

        for input_batch, output_batch in val_loader:
            input_batch = input_batch.to(device)
            examples_count += input_batch.size(0)
            output_batch = output_batch.to(device)
            output_pred_batch = model.forward(input_batch)
            loss += loss_fn(output_pred_batch, output_batch).item()

    return loss / examples_count


def validate_epoch2(
    model,
    test_loader,
    train_loader,
    loss,
    device: torch.device,
    tepoch,
):
    """
    Function to test the model, this function is called at each epoch.
    In particular, it computes the relative L^1, L^2, semi-H^1 and H^1 errors on the test set; and
    the loss on the training set with the updated parameters.

    model: the model to train
    test_loader: the test data loader (or validation loader)
    train_loader: the training data loader
    loss: the loss function that have been used during training
    exp_norm: string describing the norm used in the loss function during training
    test_samples: number of data in the test set
    training_samples: number of data in the training set
    device: the device where we have to store all the things
    tepoch: the tqdm object to print the progress
    statistic: if True, return all the loss functions, otherwise return only the same L^2 error
    """
    with torch.no_grad():
        model.eval()
        test_relative_l1 = 0.0
        test_relative_l2 = 0.0
        test_relative_semih1 = 0.0
        test_relative_h1 = 0.0
        train_loss = 0.0  # recompute the train loss with updated parameters
        training_samples_count = 0
        test_samples_count = 0

        ## Compute loss on the test set
        for input_batch, output_batch in test_loader:
            input_batch = input_batch.to(device)
            test_samples_count += input_batch.size(0)
            output_batch = output_batch.to(device)

            # compute the output
            output_pred_batch = model.forward(input_batch)

            # compute the relative L^1 error
            loss_f = LprelLoss(1, False)(output_pred_batch, output_batch)
            # loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) #!! Mishra implementation of L1 rel loss
            test_relative_l1 += loss_f.item()

            # compute the relative L^2 error
            test_relative_l2 += LprelLoss(2, False)(
                output_pred_batch, output_batch
            ).item()

            # compute the relative semi-H^1 error and H^1 error
            if model.problem_dim == 1:
                test_relative_semih1 += H1relLoss_1D(1.0, False, 0.0)(
                    output_pred_batch, output_batch
                ).item()
                test_relative_h1 += H1relLoss_1D(1.0, False)(
                    output_pred_batch, output_batch
                ).item()  # beta = 1.0 in test loss
            elif model.problem_dim == 2:
                test_relative_semih1 += H1relLoss(1.0, False, 0.0)(
                    output_pred_batch, output_batch
                ).item()
                test_relative_h1 += H1relLoss(1.0, False)(
                    output_pred_batch, output_batch
                ).item()  # beta = 1.0 in test loss

        ## Compute loss on the training set
        for input_batch, output_batch in train_loader:
            input_batch = input_batch.to(device)
            training_samples_count += input_batch.size(0)
            output_batch = output_batch.to(device)
            output_pred_batch = model(input_batch)

            loss_f = loss(output_pred_batch, output_batch)
            # loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) #!! Mishra implementation of L1 rel loss
            train_loss += loss_f.item()

        test_relative_l1 /= test_samples_count
        # test_relative_l1 /= len(test_loader) #!! For Mishra implementation
        test_relative_l2 /= test_samples_count
        test_relative_semih1 /= test_samples_count
        test_relative_h1 /= test_samples_count
        train_loss /= training_samples_count
        # train_loss /= len(train_loader) #!! For Mishra implementation

    # set the postfix for print
    tepoch.set_postfix(
        {
            "Train loss": train_loss,
            "Test rel. L^1 error": test_relative_l1,
            "Test rel. L^2 error": test_relative_l2,
            "Test rel. semi-H^1 error": test_relative_semih1,
            "Test rel. H^1 error": test_relative_h1,
        }
    )
    tepoch.close()

    return (
        test_relative_l1,
        test_relative_l2,
        test_relative_semih1,
        test_relative_h1,
        train_loss,
    )


def test_fun_multiout(
    model,
    test_loader,
    device: torch.device,
    dim_output: int,
):
    """
    As test_fun, but it returns the losses separately (one for each component of the output)
    """
    with torch.no_grad():
        model.eval()
        test_relative_l1_multiout = torch.zeros(dim_output).to(device)
        test_relative_l2_multiout = torch.zeros(dim_output).to(device)
        test_relative_semih1_multiout = torch.zeros(dim_output).to(device)
        test_relative_h1_multiout = torch.zeros(dim_output).to(device)
        test_samples_count = 0

        ## Compute loss on the test set
        for input_batch, output_batch in test_loader:
            input_batch = input_batch.to(device)
            test_samples_count += input_batch.size(0)
            output_batch = output_batch.to(device)

            # compute the output
            output_pred_batch = model.forward(input_batch)

            # compute the relative L^1 error
            loss_f = LprelLoss_multiout(1, False)(output_pred_batch, output_batch)
            test_relative_l1_multiout += loss_f

            # compute the relative L^2 error
            test_relative_l2_multiout += LprelLoss_multiout(2, False)(
                output_pred_batch, output_batch
            )

            # compute the relative semi-H^1 error and H^1 error
            if model.problem_dim == 1:
                test_relative_semih1_multiout += H1relLoss_1D_multiout(1.0, False, 0.0)(
                    output_pred_batch, output_batch
                )
                test_relative_h1_multiout += H1relLoss_1D_multiout(1.0, False)(
                    output_pred_batch, output_batch
                )  # beta = 1.0 in test loss
            elif model.problem_dim == 2:
                test_relative_semih1_multiout += H1relLoss_multiout(1.0, False, 0.0)(
                    output_pred_batch, output_batch
                )
                test_relative_h1_multiout += H1relLoss_multiout(1.0, False)(
                    output_pred_batch, output_batch
                )  # beta = 1.0 in test loss

        test_relative_l1_multiout /= test_samples_count
        test_relative_l2_multiout /= test_samples_count
        test_relative_semih1_multiout /= test_samples_count
        test_relative_h1_multiout /= test_samples_count

    return (
        test_relative_l1_multiout,
        test_relative_l2_multiout,
        test_relative_semih1_multiout,
        test_relative_h1_multiout,
    )
