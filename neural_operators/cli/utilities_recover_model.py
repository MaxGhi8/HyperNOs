"""
Utilities function for testing models.
Plotting of input/output and prediction for the trained model.
"""

import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from loss_fun import (
    H1relLoss,
    H1relLoss_1D,
    H1relLoss_1D_multiout,
    H1relLoss_multiout,
    LprelLoss,
    LprelLoss_multiout,
)
from torch import Tensor


#########################################
# Test function
#########################################
def test_fun(
    model,
    test_loader,
    train_loader,
    loss,
    problem_dim: int,
    exp_norm: str,
    device: torch.device,
    tepoch=None,
    statistic=False,
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
        test_samples = 0
        for input_batch, output_batch in test_loader:
            input_batch = input_batch.to(device)
            test_samples += input_batch.shape[0]
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
            if problem_dim == 1:
                test_relative_semih1 += H1relLoss_1D(1.0, False, 0.0)(
                    output_pred_batch, output_batch
                ).item()
                test_relative_h1 += H1relLoss_1D(1.0, False)(
                    output_pred_batch, output_batch
                ).item()  # beta = 1.0 in test loss
            elif problem_dim == 2:
                test_relative_semih1 += H1relLoss(1.0, False, 0.0)(
                    output_pred_batch, output_batch
                ).item()
                test_relative_h1 += H1relLoss(1.0, False)(
                    output_pred_batch, output_batch
                ).item()  # beta = 1.0 in test loss

        ## Compute loss on the training set
        training_samples = 0
        for input_batch, output_batch in train_loader:
            input_batch = input_batch.to(device)
            training_samples += input_batch.shape[0]
            output_batch = output_batch.to(device)
            output_pred_batch = model(input_batch)

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
    try:
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
    except Exception:
        pass

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


#########################################
# Test function with separate loss
#########################################
def test_fun_multiout(
    model,
    test_loader,
    test_samples: int,
    device: torch.device,
    dim_output: int,
    problem_dim: int,
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

        ## Compute loss on the test set
        for input_batch, output_batch in test_loader:
            input_batch = input_batch.to(device)
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
            if problem_dim == 1:
                test_relative_semih1_multiout += H1relLoss_1D_multiout(1.0, False, 0.0)(
                    output_pred_batch, output_batch
                )
                test_relative_h1_multiout += H1relLoss_1D_multiout(1.0, False)(
                    output_pred_batch, output_batch
                )  # beta = 1.0 in test loss
            elif problem_dim == 2:
                test_relative_semih1_multiout += H1relLoss_multiout(1.0, False, 0.0)(
                    output_pred_batch, output_batch
                )
                test_relative_h1_multiout += H1relLoss_multiout(1.0, False)(
                    output_pred_batch, output_batch
                )  # beta = 1.0 in test loss

        test_relative_l1_multiout /= test_samples
        test_relative_l2_multiout /= test_samples
        test_relative_semih1_multiout /= test_samples
        test_relative_h1_multiout /= test_samples

    return (
        test_relative_l1_multiout,
        test_relative_l2_multiout,
        test_relative_semih1_multiout,
        test_relative_h1_multiout,
    )


def get_tensors(model, test_loader, device: torch.device):
    """As test_fun, but it returns the tensors of the loss functions"""
    with torch.no_grad():
        model.eval()
        # initialize the tensors for IO
        input_tensor = torch.tensor([]).to(device)
        output_tensor = torch.tensor([]).to(device)
        prediction_tensor = torch.tensor([]).to(device)

        ## Compute loss on the test set
        for input_batch, output_batch in test_loader:
            input_batch = input_batch.to(device)
            input_tensor = torch.cat((input_tensor, input_batch), dim=0)

            output_batch = output_batch.to(device)
            output_tensor = torch.cat((output_tensor, output_batch), dim=0)

            # compute the output
            output_pred_batch = model.forward(input_batch)
            prediction_tensor = torch.cat((prediction_tensor, output_pred_batch), dim=0)

    return (
        input_tensor,
        output_tensor,
        prediction_tensor,
    )


def test_fun_tensors(
    model, test_loader, loss, device: torch.device, which_example: str
):
    """As test_fun, but it returns the tensors of the loss functions"""
    with torch.no_grad():
        model.eval()
        # initialize the tensors for IO
        input_tensor = torch.tensor([]).to(device)
        output_tensor = torch.tensor([]).to(device)
        prediction_tensor = torch.tensor([]).to(device)
        # initialize the tensors for losses
        test_relative_l1_tensor = torch.tensor([]).to(device)
        test_relative_l2_tensor = torch.tensor([]).to(device)
        test_relative_semih1_tensor = torch.tensor([]).to(device)
        test_relative_h1_tensor = torch.tensor([]).to(device)

        ## Compute loss on the test set
        for input_batch, output_batch in test_loader:
            input_batch = input_batch.to(device)
            input_tensor = torch.cat((input_tensor, input_batch), dim=0)

            output_batch = output_batch.to(device)
            output_tensor = torch.cat((output_tensor, output_batch), dim=0)

            # compute the output
            output_pred_batch = model.forward(input_batch)

            prediction_tensor = torch.cat((prediction_tensor, output_pred_batch), dim=0)

            # compute the relative L^1 error
            loss_f = LprelLoss(1, None)(output_pred_batch, output_batch)
            test_relative_l1_tensor = torch.cat(
                (test_relative_l1_tensor, loss_f), dim=0
            )

            # compute the relative L^1 smooth error
            # for i in range(output_pred_batch.shape[0]):
            #     loss_f = torch.nn.SmoothL1Loss()(output_pred_batch[i], output_batch[i])
            #     test_relative_l1_tensor = torch.cat(
            #         (test_relative_l1_tensor, loss_f.unsqueeze(0)), dim=0
            #     )

            # compute the relative L^2 error
            loss_f = LprelLoss(2, None)(output_pred_batch, output_batch)
            test_relative_l2_tensor = torch.cat(
                (test_relative_l2_tensor, loss_f), dim=0
            )

            # compute the relative semi-H^1 error and H^1 error
            if model.problem_dim == 1:
                loss_f = H1relLoss_1D(1.0, None, 0.0)(output_pred_batch, output_batch)
                test_relative_semih1_tensor = torch.cat(
                    (test_relative_semih1_tensor, loss_f), dim=0
                )

                loss_f = H1relLoss_1D(1.0, None)(
                    output_pred_batch, output_batch
                )  # beta = 1.0 in test loss
                test_relative_h1_tensor = torch.cat(
                    (test_relative_h1_tensor, loss_f), dim=0
                )

            elif model.problem_dim == 2:
                loss_f = H1relLoss(1.0, None, 0.0)(output_pred_batch, output_batch)
                test_relative_semih1_tensor = torch.cat(
                    (test_relative_semih1_tensor, loss_f), dim=0
                )

                loss_f = H1relLoss(1.0, None)(
                    output_pred_batch, output_batch
                )  # beta = 1.0 in test loss
                test_relative_h1_tensor = torch.cat(
                    (test_relative_h1_tensor, loss_f), dim=0
                )

    return (
        input_tensor,
        output_tensor,
        prediction_tensor,
        test_relative_l1_tensor,
        test_relative_l2_tensor,
        test_relative_semih1_tensor,
        test_relative_h1_tensor,
    )


#########################################
# Plotting poisson example
#########################################
def plot_poisson(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 12))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Source term f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approx. solution u")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Poisson equation: $-\Delta u = f$ with Dirichlet BCs")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting wave_0_5 example
#########################################
def plot_wave_0_5(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Initial condition f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u at final time T")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approx. solution u at final time T")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Wave equation")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting cont_tran example
#########################################
def plot_cont_tran(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Continuous initial data f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u at final time T")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approx. solution u at final time T")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Transport equation with continuous quantities")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting disc_tran example
#########################################
def plot_disc_tran(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Discontinuous initial data f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u at final time T")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approx. solution u at final time T")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Transport equation with discontinuous quantities")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting allen example
#########################################
def plot_allen(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Discontinuous initial data f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u at final time T")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approx. solution u at final time T")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Allen-Cahn equation")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting shear_layer example
#########################################
def plot_shear_layer(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Discontinuous initial data f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u at final time T")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approx. solution u at final time T")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Navier-Stokes equations")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting shear_layer example
#########################################
def plot_darcy(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Diffusion coefficient a")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approx. solution u")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Darcy flow equation")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting shear_layer example
#########################################
def plot_airfoil(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Initial condition")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approx. solution u")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Compressible Euler flow past airfoils")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting fhn example
#########################################
def plot_fhn(input_tensor, output_tensor, prediction_tensor, idx):
    n_points = input_tensor.shape[1]
    x_grid = torch.linspace(0, 100, n_points).to("cpu")

    # Plot the input, output and prediction for the selected samples
    fig, axs = plt.subplots(3, len(idx), figsize=(12, 4))
    for i in range(3):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                axs[i, j].plot(
                    x_grid,
                    input_tensor[idx[j], :, :].squeeze(),
                    label="Input (I_app)",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$I_app$(t)")
            elif i == 1:  # v approximation
                axs[i, j].plot(
                    x_grid, output_tensor[idx[j], :, 0].squeeze(), label="sol"
                )
                axs[i, j].plot(
                    x_grid,
                    prediction_tensor[idx[j], :, 0].squeeze(),
                    "r",
                    label="FNO",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$v$(t)")
            else:  # w approximation
                axs[i, j].plot(
                    x_grid, output_tensor[idx[j], :, 1].squeeze(), label="sol"
                )
                axs[i, j].plot(
                    x_grid,
                    prediction_tensor[idx[j], :, 1].squeeze(),
                    "r",
                    label="FNO",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$w$(t)")

            axs[i, j].set_xlabel("t")
            axs[0, j].set_ylim([0, 2])
            axs[1, j].set_ylim([-0.5, 1.5])
            axs[2, j].set_ylim([0, 2.5])
            axs[i, j].grid()
            axs[i, j].legend(loc="upper right")

    plt.suptitle("FitzHugh-Nagumo equations")  # title
    plt.tight_layout()
    plt.show()

    # Plot of the phase space for the selected samples
    fig, axs = plt.subplots(2, len(idx), figsize=(12, 4))
    for i in range(2):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                axs[i, j].plot(
                    x_grid,
                    input_tensor[idx[j], :, :].squeeze(),
                    label="Input (I_app)",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$I_app$(t)")
                axs[i, j].set_xlabel("t")
            else:  # phase space
                axs[i, j].plot(
                    output_tensor[idx[j], :, 0].squeeze(),
                    output_tensor[idx[j], :, 1].squeeze(),
                    label="sol",
                )
                axs[i, j].plot(
                    prediction_tensor[idx[j], :, 0].squeeze(),
                    prediction_tensor[idx[j], :, 1].squeeze(),
                    "r",
                    label="FNO",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$w$(t)")
                axs[i, j].set_xlabel("$v$(t)")

            axs[i, j].grid()
            axs[i, j].legend(loc="upper right")
            axs[0, j].set_ylim([0, 2])
            axs[1, j].set_xlim([-0.5, 1.5])
            axs[1, j].set_ylim([0, 2.5])

    plt.suptitle("Phase space of the FitzHugh-Nagumo equations")  # title
    plt.tight_layout()
    # plt.savefig("./fhn_examples.png", dpi=300, bbox_inches="tight")
    # plt.show()


#########################################
# Plotting hh example
#########################################
def plot_hh(input_tensor, output_tensor, prediction_tensor, idx):
    n_points = input_tensor.shape[1]
    x_grid = torch.linspace(0, 100, n_points).to("cpu")

    # Plot the input, output and prediction for the selected samples
    fig, axs = plt.subplots(5, len(idx), figsize=(12, 8))
    for i in range(5):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                axs[i, j].plot(
                    x_grid,
                    input_tensor[idx[j], :, :].squeeze(),
                    label="Input (I_app)",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$I_app$(t)")

            elif i == 1:  # v approximation
                axs[i, j].plot(
                    x_grid, output_tensor[idx[j], :, 0].squeeze(), label="sol"
                )
                axs[i, j].plot(
                    x_grid,
                    prediction_tensor[idx[j], :, 0].squeeze(),
                    "r",
                    label="FNO",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$v$(t)")

            elif i == 2:  # m approximation
                axs[i, j].plot(
                    x_grid, output_tensor[idx[j], :, 1].squeeze(), label="sol"
                )
                axs[i, j].plot(
                    x_grid,
                    prediction_tensor[idx[j], :, 1].squeeze(),
                    "r",
                    label="FNO",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$m$(t)")

            elif i == 3:  # h approximation
                axs[i, j].plot(
                    x_grid, output_tensor[idx[j], :, 2].squeeze(), label="sol"
                )
                axs[i, j].plot(
                    x_grid,
                    prediction_tensor[idx[j], :, 2].squeeze(),
                    "r",
                    label="FNO",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$h$(t)")

            elif i == 4:  # n approximation
                axs[i, j].plot(
                    x_grid, output_tensor[idx[j], :, 3].squeeze(), label="sol"
                )
                axs[i, j].plot(
                    x_grid,
                    prediction_tensor[idx[j], :, 3].squeeze(),
                    "r",
                    label="FNO",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$n$(t)")

            axs[i, j].set_xlabel("t")
            # axs[0, j].set_ylim([0, 2])
            # axs[1, j].set_ylim([-0.5, 1.5])
            # axs[2, j].set_ylim([0, 2.5])
            axs[i, j].grid()
            axs[i, j].legend(loc="upper right")

    plt.suptitle("Hodgkin-Huxley equations")  # title
    plt.tight_layout()
    # plt.savefig("./hh_examples.png", dpi=300, bbox_inches="tight")
    # plt.show()


#########################################
# Plotting OHaraRudy example
#########################################
def plot_ord(input_tensor, output_tensor, prediction_tensor, idx):
    n_points = input_tensor.shape[1]
    x_grid = torch.linspace(0, 2000, n_points).to("cpu")

    # Plot the input, output and prediction for the selected samples
    fig, axs = plt.subplots(2, len(idx), figsize=(20, 8))
    for i in range(2):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                axs[i, j].plot(
                    x_grid,
                    input_tensor[idx[j], :, :].squeeze(),
                    label="Input (I_app)",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$I_app$(t)")

            elif i == 1:  # v approximation
                axs[i, j].plot(
                    x_grid, output_tensor[idx[j], :, 11].squeeze(), label="sol"
                )
                axs[i, j].plot(
                    x_grid,
                    prediction_tensor[idx[j], :, 11].squeeze(),
                    "r",
                    label="FNO",
                )
                if j == 0:
                    axs[i, j].set_ylabel("$v$(t)")

            axs[i, j].set_xlabel("t")
            axs[i, j].grid()
            axs[i, j].legend(loc="upper right")

    plt.suptitle("OHaraRudy equations")  # title
    plt.tight_layout()
    plt.show()
    plt.savefig("figure.png")


#########################################
# Plotting crosstruss example
#########################################
def plot_crosstruss(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(7, len(idx), figsize=(16, 16))

    for i in range(7):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                colorbar.set_ticks([0, 1])
                if j == 0:
                    axs[i, j].set_ylabel("domain input")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                colorbar.set_ticks([-0.005, 0.005])
                if j == 0:
                    axs[i, j].set_ylabel("displacement x")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                colorbar.set_ticks([-0.005, 0.005])
                if j == 0:
                    axs[i, j].set_ylabel("pred disp x")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("error x")

            elif i == 4:  # output y
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 1].squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                colorbar.set_ticks([0.0, 0.1])
                if j == 0:
                    axs[i, j].set_ylabel("displacement y")

            elif i == 5:  # predicted y
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 1].squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                colorbar.set_ticks([0.0, 0.1])
                if j == 0:
                    axs[i, j].set_ylabel("pred disp y")

            elif i == 6:  # error y
                error = torch.abs(
                    output_tensor[idx[j], :, :, 1] - prediction_tensor[idx[j], :, :, 1]
                )
                im = axs[i, j].imshow(error.squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("error y")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Cross-truss structure")  # title
    plt.tight_layout()
    plt.show()
    plt.savefig("figure.png")


#########################################
# Plot for the stiffness matrix example
#########################################
def plot_stiffness_matrix(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(7, len(idx), figsize=(16, 16))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                colorbar.set_ticks([0, 1])
                if j == 0:
                    axs[i, j].set_ylabel("Domain input")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                colorbar.set_ticks([-0.005, 0.005])
                if j == 0:
                    axs[i, j].set_ylabel("Stiff matrix")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                colorbar.set_ticks([-0.005, 0.005])
                if j == 0:
                    axs[i, j].set_ylabel("Approx. stiff matrix")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Stiffness matrix")  # title
    plt.tight_layout()
    plt.show()
    plt.savefig("figure.png")


@jaxtyped(typechecker=beartype)
def test_plot_samples(
    input_tensor: Float[Tensor, "n_samples *n in_dim"],
    output_tensor: Float[Tensor, "n_samples *n d_v"],
    prediction_tensor: Float[Tensor, "n_samples *n d_v"],
    error: Float[Tensor, "n_samples"],
    mode: str,
    which_example: str,
    ntest: int,
    str_norm: str,
    n_idx: int = 5,
):
    """
    Function to plot the worst and best samples in the test set.
    mode: str
        can be "best", "worst" or "random", to plot the best, worst or random samples respectively.
    """
    error, indices = torch.sort(
        error, descending=True
    )  # Sort the error in descending order
    if mode == "worst":
        idx = indices[:n_idx].to("cpu")
        error = error[:n_idx].to("cpu")
    elif mode == "best":
        idx = indices[-n_idx:].to("cpu")
        error = error[-n_idx:].to("cpu")
    elif mode == "random":
        idx = torch.tensor(np.random.randint(0, ntest, size=(n_idx,)))
        error = error[idx].to("cpu")
    else:
        raise ValueError("The mode must be 'best', 'worst' or 'random'")

    print(f"The {mode} samples are: {idx} with error {error} in norms {str_norm}")

    match which_example:
        case "poisson":
            plot_poisson(input_tensor, output_tensor, prediction_tensor, idx)
        case "wave_0_5":
            plot_wave_0_5(input_tensor, output_tensor, prediction_tensor, idx)
        case "cont_tran":
            plot_cont_tran(input_tensor, output_tensor, prediction_tensor, idx)
        case "disc_tran":
            plot_disc_tran(input_tensor, output_tensor, prediction_tensor, idx)
        case "allen":
            plot_allen(input_tensor, output_tensor, prediction_tensor, idx)
        case "shear_layer":
            plot_shear_layer(input_tensor, output_tensor, prediction_tensor, idx)
        case "airfoil":
            plot_airfoil(input_tensor, output_tensor, prediction_tensor, idx)
        case "darcy":
            plot_darcy(input_tensor, output_tensor, prediction_tensor, idx)
        case "crosstruss":
            plot_crosstruss(input_tensor, output_tensor, prediction_tensor, idx)
        case "stiffness_matrix":
            plot_stiffness_matrix(input_tensor, output_tensor, prediction_tensor, idx)
        case "fhn":
            plot_fhn(input_tensor, output_tensor, prediction_tensor, idx)
        case "hh":
            plot_hh(input_tensor, output_tensor, prediction_tensor, idx)
        case "ord":
            plot_ord(input_tensor, output_tensor, prediction_tensor, idx)
        case _:
            raise ValueError(f"Unsupported example type: {which_example}.")
