import torch.nn as nn

class PermuteWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Input: (Batch, ..., Channels) -> (Batch, Channels, ...)
        # Move the last dimension to the second position (index 1)
        dims = list(range(x.ndim))
        dims.insert(1, dims.pop(-1))
        x = x.permute(*dims)

        x = self.model(x)

        # Output: (Batch, Channels, ...) -> (Batch, ..., Channels)
        # Move the second dimension (index 1) back to the last position
        dims = list(range(x.ndim))
        dims.append(dims.pop(1))
        x = x.permute(*dims)

        return x
