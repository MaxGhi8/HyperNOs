import torch.nn as nn


class CrossTrussWrapper(nn.Module):
    def __init__(self, model):
        super(CrossTrussWrapper, self).__init__()  # Initialize the parent class
        self.model = model

    def forward(self, input_batch):
        output_pred_batch = self.model(input_batch)
        for i in range(output_pred_batch.shape[-1]):
            output_pred_batch[:, :, :, [i]] *= input_batch
        return output_pred_batch

    def __getattr__(self, name):
        """
        Redirect attributes and methods to the wrapped model if they are not
        explicitly defined in this wrapper.
        """
        if name != "model" and hasattr(self.model, name):
            return getattr(self.model, name)
        return super().__getattr__(name)
