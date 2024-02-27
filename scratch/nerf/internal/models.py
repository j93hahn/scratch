import torch.nn as nn
from internal import fields
from internal import render


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.density_field = fields.DensityField(config)
        self.color_field = fields.ColorField(config)

    def train_step(self, xyz, delta, viewdir):
        weights, raw_features = self.density_field(xyz, delta)
        rgb = self.color_field(xyz, raw_features, viewdir)
        colors = render.get_final_image(weights, rgb)
        return colors
