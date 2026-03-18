
import torch
import torch.nn as nn
import numpy as np

class DeepXDEMIONetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.grid = None

    def forward(self, x):
        # x shape: (batch_size, width, width, in_channels)
        # DeepONet/MIONet branch input: (batch_size, input_dim)
        
        batch_size, width, _, in_channels = x.shape
        branch_input = x.view(batch_size, -1)

        # Generate grid if not already generated.
        if self.grid is None or self.grid.shape[0] != width * width:
            # DeepXDE expects grid shape: (output_dim, coordinate_dim) i.e. (width*width, 2) for 2D
            x1 = np.linspace(0, 1, width)
            x2 = np.linspace(0, 1, width)
            X1, X2 = np.meshgrid(x1, x2)
            grid = np.vstack((X1.flatten(), X2.flatten())).T # (N_points, 2)
            self.grid = torch.tensor(grid, dtype=torch.float32)
            
        # Ensure grid is on correct device
        if self.grid.device != x.device:
            self.grid = self.grid.to(x.device)
            
        trunk_input = self.grid
        
        # MIONet expects multiple branch inputs. Inputs = (branch1_in, branch2_in, trunk_in)
        # For this example, we duplicate the branch input to satisfy the requirement
        inputs = (branch_input, branch_input, trunk_input)
        out = self.model(inputs)
        
        # Output shape from DeepXDE DeepONet/MIONet: (batch, n_points)
        # We need to reshape back to (batch, width, width, out_dim)
        out_dim = getattr(self.model, "out_dim", 1)

        return out.view(batch_size, width, width, out_dim)
