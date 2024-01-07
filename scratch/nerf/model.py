"""
A model that combines Instant-NGP and Mip-NeRF 360, in the style of Nerfacto but aims to replicate
the performance of Zip-NeRF by utilizing Gaussians instead of points.
"""


import torch
import tinycudann as tcnn


class Nerfeon(torch.nn.Module):
    def __init__(self, **kwargs):   # todo: list all the parameters and their defaults
        super().__init__()




    def _instantiate_encoders(self, **kwargs):
        """
        Instantiate the density and color encodings.
        """
