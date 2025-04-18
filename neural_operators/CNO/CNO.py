import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


#########################################
# Activation Function:
#########################################
class CNO_LReLu(nn.Module):
    def __init__(self, problem_dim: int, in_size: int, out_size: int):
        super(CNO_LReLu, self).__init__()

        self.problem_dim = problem_dim
        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch channel *in_size"]
    ) -> Float[Tensor, "batch channel *out_size"]:
        if self.problem_dim == 1:
            x = F.interpolate(
                x.unsqueeze(2),
                size=(1, 2 * self.in_size),
                mode="bicubic",
                antialias=True,
            )
            x = self.act(x)
            x = F.interpolate(
                x, size=(1, self.out_size), mode="bicubic", antialias=True
            )
            return x[:, :, 0]

        elif self.problem_dim == 2:
            x = F.interpolate(
                x,
                size=(2 * self.in_size, 2 * self.in_size),
                mode="bicubic",
                antialias=True,
            )
            x = self.act(x)
            x = F.interpolate(
                x, size=(self.out_size, self.out_size), mode="bicubic", antialias=True
            )
            return x

        elif self.problem_dim == 3:
            x = F.interpolate(
                x,
                size=(2 * self.in_size, 2 * self.in_size, 2 * self.in_size),
                mode="trilinear",
                antialias=False,  # anti-aliasing not supported in 3D
            )
            x = self.act(x)
            x = F.interpolate(
                x,
                size=(self.out_size, self.out_size, self.out_size),
                mode="trilinear",
                antialias=False,  # anti-aliasing not supported in 3D
            )
            return x

        else:
            raise ValueError("Problem dimension must be 1 or 2")


#########################################
# CNO Block or Invariant Block:
#########################################
class CNOBlock(nn.Module):
    def __init__(
        self,
        problem_dim: int,
        in_channels: int,
        out_channels: int,
        in_size: int,
        out_size: int,
        kernel_size: int = 3,
        use_bn: bool = False,
    ):
        super(CNOBlock, self).__init__()

        self.problem_dim = problem_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.use_bn = use_bn

        if self.problem_dim == 1:
            self.convolution = torch.nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=not self.use_bn,
            )

            if self.use_bn:
                self.batch_norm = nn.BatchNorm1d(self.out_channels)
            else:
                self.batch_norm = nn.Identity()

        elif self.problem_dim == 2:
            self.convolution = torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=not self.use_bn,
            )

            if use_bn:
                self.batch_norm = nn.BatchNorm2d(self.out_channels)
            else:
                self.batch_norm = nn.Identity()

        elif self.problem_dim == 3:
            self.convolution = torch.nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=not self.use_bn,
            )

            if use_bn:
                self.batch_norm = nn.BatchNorm3d(self.out_channels)
            else:
                self.batch_norm = nn.Identity()

        else:
            raise ValueError("Problem dimension must be 1, 2 or 3")

        # Up/Down-sampling happens inside Activation
        self.act = CNO_LReLu(
            self.problem_dim, in_size=self.in_size, out_size=self.out_size
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch {self.in_channels} *in_size"]
    ) -> Float[Tensor, "batch {self.out_channels} *out_size"]:
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)


#########################################
# Lift/Project Block:
#########################################
class LiftProjectBlock(nn.Module):
    def __init__(
        self, problem_dim, in_channels, out_channels, size, latent_dim=64, kernel_size=3
    ):
        super(LiftProjectBlock, self).__init__()

        self.problem_dim = problem_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2

        self.inter_CNOBlock = CNOBlock(
            problem_dim=self.problem_dim,
            in_channels=self.in_channels,
            out_channels=self.latent_dim,
            in_size=self.size,
            out_size=self.size,
            kernel_size=self.kernel_size,
            use_bn=False,
        )

        if self.problem_dim == 1:
            self.convolution = torch.nn.Conv1d(
                in_channels=self.latent_dim,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
        elif self.problem_dim == 2:
            self.convolution = torch.nn.Conv2d(
                in_channels=self.latent_dim,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
        elif self.problem_dim == 3:
            self.convolution = torch.nn.Conv3d(
                in_channels=self.latent_dim,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
        else:
            raise ValueError("Problem dimension must be 1, 2 or 3")

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch {self.in_channels} *size"]
    ) -> Float[Tensor, "batch {self.out_channels} *size"]:
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        return x


#########################################
# Residual Block:
#########################################
class ResidualBlock(nn.Module):
    def __init__(self, problem_dim, channels, size, kernel_size=3, use_bn=True):
        super(ResidualBlock, self).__init__()

        self.problem_dim = problem_dim
        self.channels = channels
        self.size = size
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.use_bn = use_bn

        if self.problem_dim == 1:
            self.convolution1 = torch.nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=not self.use_bn,
            )
            self.convolution2 = torch.nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=not self.use_bn,
            )

            if self.use_bn:
                self.batch_norm1 = nn.BatchNorm1d(self.channels)
                self.batch_norm2 = nn.BatchNorm1d(self.channels)

            else:
                self.batch_norm1 = nn.Identity()
                self.batch_norm2 = nn.Identity()

        elif self.problem_dim == 2:
            self.convolution1 = torch.nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=not self.use_bn,
            )
            self.convolution2 = torch.nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=not self.use_bn,
            )

            if self.use_bn:
                self.batch_norm1 = nn.BatchNorm2d(self.channels)
                self.batch_norm2 = nn.BatchNorm2d(self.channels)

            else:
                self.batch_norm1 = nn.Identity()
                self.batch_norm2 = nn.Identity()

        elif self.problem_dim == 3:
            self.convolution1 = torch.nn.Conv3d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=not self.use_bn,
            )
            self.convolution2 = torch.nn.Conv3d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=not self.use_bn,
            )

            if self.use_bn:
                self.batch_norm1 = nn.BatchNorm3d(self.channels)
                self.batch_norm2 = nn.BatchNorm3d(self.channels)

            else:
                self.batch_norm1 = nn.Identity()
                self.batch_norm2 = nn.Identity()

        else:
            raise ValueError("Problem dimension must be 1, 2 or 3")

        # Up/Down-sampling happens inside Activation
        self.act = CNO_LReLu(self.problem_dim, in_size=self.size, out_size=self.size)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch {self.channels} *in_size"]
    ) -> Float[Tensor, "batch {self.channels} *out_size"]:
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out


#########################################
# ResNet:
#########################################
class ResNet(nn.Module):
    def __init__(
        self, problem_dim, channels, size, num_blocks, kernel_size=3, use_bn=True
    ):
        super(ResNet, self).__init__()

        self.problem_dim = problem_dim
        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.use_bn = use_bn

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(
                ResidualBlock(
                    problem_dim=problem_dim,
                    channels=channels,
                    size=size,
                    kernel_size=self.kernel_size,
                    use_bn=use_bn,
                )
            )

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch {self.channels} *in_size"]
    ) -> Float[Tensor, "batch {self.channels} *out_size"]:
        for i in range(self.num_blocks):
            x = self.res_nets[i](x)
        return x


#########################################
# CNO:
#########################################
class CNO(nn.Module):
    def __init__(
        self,
        problem_dim,
        in_dim,
        out_dim,
        size,
        N_layers,
        N_res=4,
        N_res_neck=4,
        channel_multiplier=16,
        kernel_size=3,
        use_bn=True,
        device=torch.device("cpu"),
    ):
        """
        CNO: Convolutional Neural Operator

        Parameters:

        in_dim: int
            Number of input channels.

        out_dim: int
            Number of output channels.

        size: int
            Input and Output spatial size

        N_layers: int
            Number of (D) or (U) blocks in the network

        N_res: int
            Number of (R) blocks per level (except the neck)

        N_res_neck: int
            Number of (R) blocks in the neck

        channel_multiplier: int
            multiplier of the number of channels in the network

        kernel_size: int
            size of the convolutional kernel

        use_bn: bool
            choose if Batch Normalization is used.
        """
        super(CNO, self).__init__()

        self.problem_dim = problem_dim
        self.N_layers = N_layers
        self.lift_dim = (
            channel_multiplier // 2
        )  # Input is lifted to the half of channel_multiplier dimension
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel_multiplier = channel_multiplier  # The growth of the channels
        self.kernel_size = kernel_size
        self.use_bn = use_bn

        #### Num of channels/features - evolution
        # Encoder (at every layer the number of channels is doubled)
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append((2**i) * self.channel_multiplier)

        # Decoder (at every layer the number of channels is halved)
        self.decoder_features_in = self.encoder_features[
            1:
        ]  # How the features in Decoder evolve (number of features)
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = (
                2 * self.decoder_features_in[i]
            )  # Concat the outputs of the res-nets, so we must multiply by 2

        #### Spatial sizes of channels (grid resolution) - evolution
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(
                size // (2**i)
            )  # Encoder sizes are halved at every layer
            self.decoder_sizes.append(
                size // (2 ** (self.N_layers - i))
            )  # Decoder sizes are doubled at every layer

        #### Define Lift and Project blocks
        self.lift = LiftProjectBlock(
            problem_dim=self.problem_dim,
            in_channels=in_dim,
            out_channels=self.encoder_features[0],
            size=size,
            kernel_size=self.kernel_size,
        )

        self.project = LiftProjectBlock(
            problem_dim=self.problem_dim,
            in_channels=self.encoder_features[0]
            + self.decoder_features_out[-1],  # concatenation with the ResNet
            out_channels=out_dim,
            size=size,
            kernel_size=self.kernel_size,
        )

        #### Define Encoder, ED Linker and Decoder networks
        self.encoder = nn.ModuleList(
            [
                CNOBlock(
                    problem_dim=self.problem_dim,
                    in_channels=self.encoder_features[i],
                    out_channels=self.encoder_features[i + 1],
                    in_size=self.encoder_sizes[i],
                    out_size=self.encoder_sizes[i + 1],
                    kernel_size=self.kernel_size,
                    use_bn=self.use_bn,
                )
                for i in range(self.N_layers)
            ]
        )

        # After the ResNets are executed, the sizes of encoder and decoder might not match
        # We must ensure that the sizes are the same, by applying CNO Blocks
        self.ED_expansion = nn.ModuleList(
            [
                CNOBlock(
                    problem_dim=self.problem_dim,
                    in_channels=self.encoder_features[i],
                    out_channels=self.encoder_features[i],
                    in_size=self.encoder_sizes[i],
                    out_size=self.decoder_sizes[self.N_layers - i],
                    kernel_size=self.kernel_size,
                    use_bn=use_bn,
                )
                for i in range(self.N_layers + 1)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                CNOBlock(
                    problem_dim=self.problem_dim,
                    in_channels=self.decoder_features_in[i],
                    out_channels=self.decoder_features_out[i],
                    in_size=self.decoder_sizes[i],
                    out_size=self.decoder_sizes[i + 1],
                    kernel_size=self.kernel_size,
                    use_bn=use_bn,
                )
                for i in range(self.N_layers)
            ]
        )

        self.inv_features = self.decoder_features_in
        self.inv_features.append(
            self.encoder_features[0] + self.decoder_features_out[-1]
        )
        self.decoder_inv = nn.ModuleList(
            [
                CNOBlock(
                    problem_dim=self.problem_dim,
                    in_channels=self.inv_features[i],
                    out_channels=self.inv_features[i],
                    in_size=self.decoder_sizes[i],
                    out_size=self.decoder_sizes[i],
                    kernel_size=self.kernel_size,
                    use_bn=use_bn,
                )
                for i in range(self.N_layers + 1)
            ]
        )

        #### Define ResNets Blocks
        self.res_nets = nn.ModuleList()
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet networks (before the neck)
        for i in range(self.N_layers):
            self.res_nets.append(
                ResNet(
                    problem_dim=self.problem_dim,
                    channels=self.encoder_features[i],
                    size=self.encoder_sizes[i],
                    num_blocks=self.N_res,
                    kernel_size=self.kernel_size,
                    use_bn=self.use_bn,
                )
            )

        self.res_net_neck = ResNet(
            problem_dim=self.problem_dim,
            channels=self.encoder_features[self.N_layers],
            size=self.encoder_sizes[self.N_layers],
            num_blocks=self.N_res_neck,
            kernel_size=self.kernel_size,
            use_bn=self.use_bn,
        )

        # Move to device
        self.to(device)
        self.device = device

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch *size {self.in_dim}"]
    ) -> Float[Tensor, "batch *size {self.out_dim}"]:

        if self.problem_dim == 1:
            x = self.lift(x.permute(0, 2, 1))  # Execute Lift
        elif self.problem_dim == 2:
            x = self.lift(x.permute(0, 3, 1, 2))  # Execute Lift
        elif self.problem_dim == 3:
            x = self.lift(x.permute(0, 4, 1, 2, 3))
        else:
            raise ValueError("Problem dimension must be 1 or 2")

        skip = []

        # Execute Encoder
        for i in range(self.N_layers):
            # Apply ResNet and save the result
            y = self.res_nets[i](x)
            skip.append(y)

            # Apply (D) block
            x = self.encoder[i](x)

        # Apply the deepest ResNet (bottle neck)
        x = self.res_net_neck(x)

        # Execute Decode
        for i in range(self.N_layers):
            # Apply (I) block (ED_expansion) and cat if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x)  # BottleNeck : no cat
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])), 1)

            x = self.decoder_inv[i](x)

            # Apply (U) block
            x = self.decoder[i](x)

        # Cat and Execute Projection
        x = torch.cat((x, self.ED_expansion[0](skip[0])), 1)

        if self.problem_dim == 1:
            return self.project(x).permute(0, 2, 1)
        elif self.problem_dim == 2:
            # project and reshape (for consistency with the rest of the models)
            return self.project(x).permute(0, 2, 3, 1)
        elif self.problem_dim == 3:
            # project and reshape (for consistency with the rest of the models)
            return self.project(x).permute(0, 2, 3, 4, 1)
        else:
            raise ValueError("Problem dimension must be 1, 2 or 3")
