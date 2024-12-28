import torch.nn as nn


class CrossTrussWrapper(nn.Module):
    def __init__(self, model):
        self.model = model

    def forward(self, input_batch):
        output_pred_batch = self.model(input_batch)
        for i in range(output_pred_batch.shape[-1]):
            output_pred_batch[:, :, :, [i]] *= input_batch
        return output_pred_batch
