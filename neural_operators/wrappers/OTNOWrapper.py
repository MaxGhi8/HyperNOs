import torch
import torch.nn as nn

class OTNOWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Input: (Batch, ..., Channels) -> (Batch, Channels, ...)
        # Move the last dimension to the second position (index 1)
        dims = list(range(x.ndim))
        dims.insert(1, dims.pop(-1))
        x = x.permute(*dims)
        
        # Store original spatial shape for reshaping output
        # x is (Batch, Channels, H, W)
        original_shape = x.shape
        batch_size = x.shape[0]
        # spatial dims are from index 2 onwards
        spatial_shape = x.shape[2:] 
        
        # Create identity ind_dec (flattened grid)
        # Assuming n_s = H*W
        n_s = 1
        for d in spatial_shape:
            n_s *= d
            
        ind_dec = torch.arange(n_s, device=x.device)

        # Iterate over batch because OTNO model does not handle batch dimension correctly
        outputs = []
        for i in range(batch_size):
             # Unsqueeze to add batch dim back for lifting (model expects (1, C, H, W) effectively or handles (C, H, W)?)
             # otno.py forward: x = self.lifting(x). lifting is channel mlp or linear.
             
             # Let's pass (1, C, ...) 
             x_i = x[i].unsqueeze(0) 
             out_i = self.model(x_i, ind_dec=ind_dec)
             outputs.append(out_i)
        
        # Stack outputs: (Batch, OutChannels, n_s)
        x = torch.cat(outputs, dim=0)
        
        # Output from OTNO (collected) is (Batch, OutChannels, n_t) where n_t = n_s
        # Reshape back to spatial grid
        # (Batch, OutChannels, H, W)
        x = x.reshape(batch_size, -1, *spatial_shape)

        # Output: (Batch, Channels, ...) -> (Batch, ..., Channels)
        # Move the second dimension (index 1) back to the last position
        dims = list(range(x.ndim))
        dims.append(dims.pop(1))
        x = x.permute(*dims)

        return x
