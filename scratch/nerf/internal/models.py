import torch
import torch.nn as nn
import gin


@gin.configurable
class Model(nn.Module):
    def __init__(self, encoder, color_field):
        super().__init__()
        self.encoder = encoder
        self.color_field = color_field

    def forward(self, xyz, delta, viewdir):
        raw = self.encoder(xyz)
