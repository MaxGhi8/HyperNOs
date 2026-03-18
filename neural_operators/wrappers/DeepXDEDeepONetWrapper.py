import torch
import torch.nn as nn

class DeepXDEDeepONetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model= model
        self.grid = None
        self.cached_width = -1

    def forward(self, x):
        # x shape: (batch_size, width, width, in_channels)
        # DeepONet branch input: (batch_size, input_dim)
        batch_size = x.shape[0]
        width = x.shape[1]
        
        # Check if we need to update the grid
        if width != self.cached_width or self.grid is None:
            self.cached_width = width

            # Assuming normalized [0,1]x[0,1] grid
            start = 0
            end = 1
            vals = torch.linspace(start, end, width, device=x.device)
            grid_x, grid_y = torch.meshgrid(vals, vals, indexing="ij")
            
            # Flatten grid to (N_points, 2)
            self.grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        # x is (B, W, W, C) -> (B, W*W*C)
        branch_input = x.view(batch_size, -1)
        
        # Trunk input is the grid, constant for all samples
        # shape: (width*width, 2)
        trunk_input = self.grid
        
        # Ensure grid is on the same device as input
        if trunk_input.device != x.device:
            trunk_input = trunk_input.to(x.device)
            self.grid = trunk_input
        
        # Forward pass
        out = self.model((branch_input, trunk_input))
        
        # Reshape output back to (batch_size, width, width, out_dim)
        out = out.view(batch_size, width, width, self.model.out_dim)
        
        return out
