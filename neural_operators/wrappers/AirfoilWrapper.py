import torch.nn as nn


class AirfoilWrapper(nn.Module):
    def __init__(self, model):
        super(AirfoilWrapper, self).__init__()  # Initialize the parent class
        self.model = model

    def forward(self, input_batch):
        output_batch = self.model(input_batch)
        output_batch[input_batch == 1] = 1
        return output_batch
