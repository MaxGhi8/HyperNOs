import torch
import torch.nn as nn


class DeepONetWrapper(nn.Module):
    """
    Wrapper for DeepONet that reshapes the output from (batch_size, n_points, n_output*)
    to (batch_size, n, n, n_output) assuming the points come from a regular 2D grid.
    """

    def __init__(self, model, grid_size: int):
        super(DeepONetWrapper, self).__init__()
        self.model = model
        self.grid_size = grid_size
        self.n_points = grid_size * grid_size

    def forward(self, input_batch):
        output = self.model(input_batch)
        batch_size = output.shape[0]
        output = output.view(batch_size, self.grid_size, self.grid_size, self.n_output)

        return output

    def __getattr__(self, name):
        """
        Redirect attributes and methods to the wrapped model if they are not
        explicitly defined in this wrapper.
        """
        if (
            name != "model"
            and name != "grid_size"
            and name != "n_points"
            and hasattr(self.model, name)
        ):
            return getattr(self.model, name)
        return super().__getattr__(name)
