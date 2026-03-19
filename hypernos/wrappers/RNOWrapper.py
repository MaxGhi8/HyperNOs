"""
Since RNO expects a time dimension (it's a recurrent model) and standard datasets like "poisson"
 are static 2D, this wrapper transforms the input (Batch, Channels, H, W) into 
 (Batch, 1, Channels, H, W), effectively treating it as a time sequence of length 1. 
 This allows RNO to work on your existing data.
"""
import torch.nn as nn

class RNOWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Input: (Batch, H, W, Channels) - assumes 2D spatial + channels last
        # We need to reshape to (Batch, 1, Channels, H, W)
        
        # 1. Permute to (Batch, Channels, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # 2. Add time dimension
        x = x.unsqueeze(1)
        
        # 3. Pass to model
        x = self.model(x)
        # Output is (Batch, OutChannels, H, W)
        
        # 4. Permute back to (Batch, H, W, OutChannels)
        x = x.permute(0, 2, 3, 1)
        return x
