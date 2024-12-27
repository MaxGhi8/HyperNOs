import os
import tempfile

import torch
from Loss_fun import H1relLoss, H1relLoss_1D, LprelLoss
from ray import train
from ray.train import Checkpoint


def train_model(config, model, train_loader, val_loader, loss_fn, device):
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

    for ep in range(start_epoch, config["epochs"]):
        # Train the model for one epoch
        train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)

        # Validate the model for one epoch
        acc = validate_epoch(model, val_loader, train_loader, loss_fn, device)

        if ep % config["checkpoint_frequency"] == 0 or ep == config["epochs"] - 1:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report({"relative_loss": acc}, checkpoint=checkpoint)
        else:
            train.report({"relative_loss": acc})


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
        # TODO: adjust models for airfoil and crosstruss
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


# TODO: implement
def validate_epoch(
    model,
    test_loader,
    train_loader,
    loss,
    device: torch.device,
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
    which_example: the example of the PDEs that we are considering
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

        ## Compute loss on the test set
        for input_batch, output_batch in test_loader:
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            # compute the output
            output_pred_batch = model.forward(input_batch)

            if which_example == "airfoil":
                output_pred_batch[input_batch == 1] = 1
                output_batch[input_batch == 1] = 1
            elif which_example == "crosstruss":
                for i in range(input_batch.shape[-1]):
                    output_pred_batch[:, :, :, [i]] = (
                        output_pred_batch[:, :, :, [i]] * input_batch
                    )
                    output_batch[:, :, :, [i]] = (
                        output_batch[:, :, :, [i]] * input_batch
                    )

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
            output_batch = output_batch.to(device)
            output_pred_batch = model(input_batch)

            if which_example == "airfoil":
                output_pred_batch[input_batch == 1] = 1
                output_batch[input_batch == 1] = 1
            elif which_example == "crosstruss":
                for i in range(input_batch.shape[-1]):
                    output_pred_batch[:, :, :, [i]] = (
                        output_pred_batch[:, :, :, [i]] * input_batch
                    )
                    output_batch[:, :, :, [i]] = (
                        output_batch[:, :, :, [i]] * input_batch
                    )

            loss_f = loss(output_pred_batch, output_batch)
            # loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) #!! Mishra implementation of L1 rel loss
            train_loss += loss_f.item()

        test_relative_l1 /= test_samples
        # test_relative_l1 /= len(test_loader) #!! For Mishra implementation
        test_relative_l2 /= test_samples
        test_relative_semih1 /= test_samples
        test_relative_h1 /= test_samples
        train_loss /= training_samples
        # train_loss /= len(train_loader) #!! For Mishra implementation

    # set the postfix for print
    if tepoch is not None:
        tepoch.set_postfix(
            {
                "Train loss " + exp_norm: train_loss,
                "Test rel. L^1 error": test_relative_l1,
                "Test rel. L^2 error": test_relative_l2,
                "Test rel. semi-H^1 error": test_relative_semih1,
                "Test rel. H^1 error": test_relative_h1,
            }
        )
        tepoch.close()

    if statistic:
        return (
            test_relative_l1,
            test_relative_l2,
            test_relative_semih1,
            test_relative_h1,
            train_loss,
        )
    else:
        match exp_norm:
            case "L1":
                return test_relative_l1
            case "L2":
                return test_relative_l2
            case "H1":
                return test_relative_h1
            case _:
                raise ValueError("The norm is not implemented")
