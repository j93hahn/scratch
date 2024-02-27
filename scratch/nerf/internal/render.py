import torch
import torch.nn as nn


def get_final_image(weights, quantity):
    """
    a general purpose function to compute the final image from the weights and the quantity

    supports rgb, density, and normals
    """
    image = torch.sum(weights * quantity, dim=-1)
    return image


class Rays(nn.Module):
    """
    ray class to store ray information that will be used in sampling, rendering,
    and volumetric integration
    """
    def __init__(self):
        self.origins = None
        self.directions = None
        self.viewdirs = None
        self.t_vals = None
        self.mins = None
        self.maxs = None
        self.near = None
        self.far = None
