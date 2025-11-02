import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)


class FeedForward(nn.Module):
    def __init__(self, n_input, n_output, hidden, act_fun, output_act=False):
        super().__init__()
        self.to(device, dtype=torch.float32)
        self.n_input = n_input
        self.n_output = n_output
        self.hidden = hidden
        self.act_fun = act_fun
        self.output_act = output_act
        self.n_hidden_layers = len(hidden)

        self.NN_list = nn.ModuleList()
        self.NN_list.append(nn.Linear(self.n_input, self.hidden[0]))
        # nn.init.xavier_normal_(self.NN_list[0].weight)
        # nn.init.zeros_(self.NN_list[0].bias)

        for i in range(self.n_hidden_layers - 1):

            self.NN_list.append(nn.Linear(self.hidden[i], self.hidden[i + 1]))
            # nn.init.xavier_normal_(self.NN_list[i].weight)
            # nn.init.zeros_(self.NN_list[i].bias)

        self.NN_list.append(nn.Linear(self.hidden[-1], self.n_output))
        # nn.init.xavier_normal_(self.NN_list[-1].weight)

    def forward(self, x):
        x = x.to(device=device, dtype=torch.float32)
        for i in range(self.n_hidden_layers):
            x = self.NN_list[i](x)
            x = self.act_fun(x)
        x = self.NN_list[-1](x)
        if self.output_act == True:
            x = self.act_fun(x)

        return x


class Residual_FeedForward(nn.Module):
    def __init__(self, n_input, n_output, hidden, act_fun, output_act=False):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.hidden = hidden
        self.act_fun = act_fun
        self.output_act = output_act
        self.n_hidden_layers = len(hidden)

        self.NN_list = nn.ModuleList()
        self.NN_list.append(nn.Linear(self.n_input, self.hidden[0]))
        # nn.init.xavier_normal_(self.NN_list[0].weight)
        # nn.init.zeros_(self.NN_list[0].bias)

        for i in range(self.n_hidden_layers - 1):

            self.NN_list.append(nn.Linear(self.hidden[i], self.hidden[i + 1]))
            # nn.init.xavier_normal_(self.NN_list[i].weight)
            # nn.init.zeros_(self.NN_list[i].bias)

        self.NN_list.append(nn.Linear(self.hidden[-1], self.n_output))
        # nn.init.xavier_normal_(self.NN_list[-1].weight)
        self.to(device, dtype=torch.float32)

    def forward(self, x):

        x = x.to(device=device, dtype=torch.float32)
        for i in range(self.n_hidden_layers):
            res = x
            x = self.NN_list[i](x)
            if x.shape == res.shape:
                x = x + res
            else:
                x = self.act_fun(x)
        x = self.NN_list[-1](x)
        if self.output_act == True:
            x = self.act_fun(x)

        return x


class branch_2D(nn.Module):
    def __init__(
        self,
        n_input,
        conv_layers,
        stride,
        kernel_size,
        hidden_layer,
        n_basis,
        act_fun,
        output_dim_conv=1,
        n_points=100,
    ):
        super().__init__()
        self.n_input_branch = n_input
        self.conv_layers = conv_layers
        self.stride = stride
        self.kernel_size = kernel_size
        self.hidden_layer = hidden_layer
        self.n_basis = n_basis
        self.act_fun = act_fun
        self.output_dim_conv = output_dim_conv
        self.n_points = n_points

        self.NN_list = nn.ModuleList()
        self.NN_list.append(
            nn.Conv2d(
                in_channels=self.n_input_branch,
                out_channels=self.conv_layers[0],
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=self.stride,
                padding="valid",
            )
        )

        for index in range(len(self.conv_layers) - 1):
            self.NN_list.append(
                nn.Conv2d(
                    in_channels=self.conv_layers[index],
                    out_channels=self.conv_layers[index + 1],
                    kernel_size=(self.kernel_size, self.kernel_size),
                    stride=self.stride,
                    padding="valid",
                )
            )

        if self.output_dim_conv != 1:
            self.NN_list.append(
                nn.Conv2d(
                    in_channels=self.conv_layers[-1],
                    out_channels=self.conv_layers[-1],
                    kernel_size=(self.output_dim_conv, self.output_dim_conv),
                    stride=1,
                )
            )

        self.NN_list.append(nn.Flatten())

        self.to(device, dtype=torch.float32)

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.n_input_branch,
                n_points,
                n_points,
                device=device,
                dtype=torch.float32,
            )
            for layer in self.NN_list:
                if isinstance(layer, nn.Flatten):
                    dummy = layer(dummy)
                    break
                dummy = layer(dummy)
                dummy = self.act_fun(dummy)
            flattened_dim = dummy.view(1, -1).shape[1]

        self.NN_list.append(nn.Linear(flattened_dim, self.hidden_layer[0]))
        for index in range(len(self.hidden_layer) - 1):
            self.NN_list.append(
                nn.Linear(self.hidden_layer[index], self.hidden_layer[index + 1]),
            )
        self.NN_list.append(nn.Linear(self.hidden_layer[-1], self.n_basis))

    def forward(self, x):
        x = x.to(device=device, dtype=torch.float32)
        for i in range(len(self.NN_list) - 1):
            x = self.NN_list[i](x)
            x = self.act_fun(x)
        x = self.NN_list[-1](x)
        return x


class residual_branch_2D(nn.Module):
    def __init__(
        self,
        n_input,
        conv_layers,
        stride,
        kernel_size,
        hidden_layer,
        n_basis,
        act_fun,
        output_dim_conv=1,
        n_points=100,
    ):
        super().__init__()

        self.n_input_branch = n_input
        self.conv_layers = conv_layers
        self.stride = stride
        self.kernel_size = kernel_size
        self.hidden_layer = hidden_layer
        self.n_basis = n_basis
        self.act_fun = act_fun
        self.output_dim_conv = output_dim_conv
        self.n_points = n_points

        self.NN_list = nn.ModuleList()
        self.NN_list.append(
            nn.Conv2d(
                in_channels=self.n_input_branch,
                out_channels=self.conv_layers[0],
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=self.stride,
                padding="valid",
            )
        )

        for index in range(len(self.conv_layers) - 1):
            self.NN_list.append(
                nn.Conv2d(
                    in_channels=self.conv_layers[index],
                    out_channels=self.conv_layers[index + 1],
                    kernel_size=(self.kernel_size, self.kernel_size),
                    stride=self.stride,
                    padding="valid",
                )
            )

        if self.output_dim_conv != 1:
            self.NN_list.append(
                nn.Conv2d(
                    in_channels=self.conv_layers[-1],
                    out_channels=self.conv_layers[-1],
                    kernel_size=(self.output_dim_conv, self.output_dim_conv),
                    stride=1,
                )
            )

        self.NN_list.append(nn.Flatten())
        self.to(device, dtype=torch.float32)

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.n_input_branch,
                n_points,
                n_points,
                device=device,
                dtype=torch.float32,
            )
            for layer in self.NN_list:
                if isinstance(layer, nn.Flatten):
                    dummy = layer(dummy)
                    break
                dummy = layer(dummy)
                dummy = self.act_fun(dummy)
            flattened_dim = dummy.view(1, -1).shape[1]
        self.NN_list.append(nn.Linear(flattened_dim, self.hidden_layer[0]))
        for index in range(len(self.hidden_layer) - 1):
            self.NN_list.append(
                nn.Linear(self.hidden_layer[index], self.hidden_layer[index + 1]),
            )
        self.NN_list.append(nn.Linear(self.hidden_layer[-1], self.n_basis))

    def forward(self, x):
        x = x.to(device=device, dtype=torch.float32)
        for i in range(len(self.NN_list) - 1):
            res = x
            x = self.NN_list[i](x)
            if x.shape == res.shape:
                x = x + res
            else:
                x = self.act_fun(x)
        x = self.NN_list[-1](x)
        return x
