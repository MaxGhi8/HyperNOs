import torch
import torch.nn as nn


class DeepONetWrapper(nn.Module):
    """
    Wrapper for DeepONet that reshapes the output from (batch_size, n_points)
    to (batch_size, n, n) assuming the points come from a regular 2D grid.
    """

    def __init__(self, model, grid_size: int):
        super(DeepONetWrapper, self).__init__()
        self.model = model
        self.grid_size = grid_size
        self.n_points = grid_size * grid_size

    def forward(self, input_batch):
        """
        Args:
            input_batch: Tuple of (branch_input, trunk_input)

        Returns:
            output: Tensor of shape (batch_size, grid_size, grid_size, 1) for single output
                    or (batch_size, grid_size, grid_size, n_output) for multiple outputs
        """
        output = self.model(input_batch)

        assert (
            output.shape[-1] == self.n_points
        ), f"Expected {self.n_points} points, got {output.shape[1]}"

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
