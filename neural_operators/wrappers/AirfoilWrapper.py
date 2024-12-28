import torch.nn as nn


class AirfoilWrapper(nn.Module):
    def __init__(self, model):
        super(AirfoilWrapper, self).__init__()  # Initialize the parent class
        self.model = model

    def forward(self, input_batch):
        output_batch = self.model(input_batch)
        output_batch[input_batch == 1] = 1
        return output_batch

    def __getattr__(self, name):
        """
        Redirect attributes and methods to the wrapped model if they are not
        explicitly defined in this wrapper.
        """
        if name != "model" and hasattr(self.model, name):
            return getattr(self.model, name)
        return super().__getattr__(name)
