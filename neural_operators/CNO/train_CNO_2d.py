import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import AdamW

# from CNO_2d import CNO2d
from CNO2d import CNO2d

from CNO_benchmarks import SinFrequency
from CNO_utilities import CNO_initialize_hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def main():

    # --------------------------------------
    # REPLACE THIS PART BY YOUR DATALOADER
    # --------------------------------------

    n_train = 100  # number of training samples

    x_data = torch.rand((256, 1, 128, 128)).type(torch.float32)
    y_data = torch.ones((256, 1, 128, 128)).type(torch.float32)

    input_function_train = x_data[:n_train, :]
    output_function_train = y_data[:n_train, :]
    input_function_test = x_data[n_train:, :]
    output_function_test = y_data[n_train:, :]

    batch_size = 10

    training_set = DataLoader(
        TensorDataset(input_function_train, output_function_train),
        batch_size=batch_size,
        shuffle=True,
    )
    testing_set = DataLoader(
        TensorDataset(input_function_test, output_function_test),
        batch_size=batch_size,
        shuffle=False,
    )

    # cno_architecture = CNO_initialize_hyperparameters("poisson", "default")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # example = SinFrequency(cno_architecture, device, 32, 126, in_dist=True, search_path='/')

    # ---------------------
    # Define the hyperparameters and the model:
    # ---------------------

    learning_rate = 0.001
    epochs = 50
    step_size = 15
    gamma = 0.5

    N_layers = 3
    N_res = 6
    N_res_neck = 2
    channel_multiplier = 32

    s = 128

    cno = CNO2d(
        in_dim=1,  # Number of input channels.
        out_dim=1,  # Number of input channels.
        size=s,  # Input and Output spatial size (required )
        N_layers=N_layers,  # Number of (D) or (U) blocks in the network
        N_res=N_res,  # Number of (R) blocks per level (except the neck)
        N_res_neck=N_res_neck,  # Number of (R) blocks in the neck
        channel_multiplier=channel_multiplier,  # How the number of channels evolve?
        use_bn=False,
    )

    cno.to(device)
    # -----------
    # TRAIN:
    # -----------

    nparams = 0
    for param in cno.parameters():
        nparams += param.numel()
    print(f"Total number of model parameters: {nparams}")

    optimizer = AdamW(cno.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    l = nn.L1Loss()
    freq_print = 1
    for epoch in range(epochs):
        train_mse = 0.0
        for input_batch, output_batch in training_set:
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            optimizer.zero_grad()
            output_pred_batch = cno(input_batch)
            loss_f = l(output_pred_batch, output_batch)
            loss_f.backward()
            optimizer.step()
            train_mse += loss_f.item()
        train_mse /= len(training_set)

        scheduler.step()

        with torch.no_grad():
            cno.eval()
            test_relative_l2 = 0.0
            for input_batch, output_batch in testing_set:
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)

                output_pred_batch = cno(input_batch)
                loss_f = (
                    torch.mean((abs(output_pred_batch - output_batch)))
                    / torch.mean(abs(output_batch))
                ) ** 0.5
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(testing_set)

        if epoch % freq_print == 0:
            print(
                "######### Epoch:",
                epoch,
                " ######### Train Loss:",
                train_mse,
                " ######### Relative L1 Test Norm:",
                test_relative_l2,
            )


if __name__ == "__main__":
    main()
