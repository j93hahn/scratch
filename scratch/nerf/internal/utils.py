import numpy as np
import torch
import torch.nn as nn


def fourier_features(x, B):
    """
    applies fourier feature mapping to the input x; B is sampled from a Gaussian distribution

    input arguments:
        - x has shape [..., d] where d is the number of features
        - B has shape [m, d] where m is the number of fourier features

    returns:
        x_proj has shape [..., 2m]
    """
    x_proj = x @ B.T
    x_proj = 2 * np.pi * x_proj
    x_proj = torch.cat(
        [torch.sin(x_proj), torch.cos(x_proj)],
        dim=-1
    )
    assert x_proj.shape[-1] == 2 * B.shape[0] and x_proj.shape[:-1] == x.shape[:-1], \
        f"Expected shape [..., {2 * B.shape[0]}], got {x_proj.shape}"
    return x_proj


class SceneContraction(nn.Module):
    """
    applies scene contraction to the input positions

    by default, the order of contraction is set to 'inf' which corresponds to the max norm.
    it's useful in the case of a hash grid encoder
    """
    def __init__(self, order='inf'):
        super().__init__()
        self.order = order

    def contract(self, x):
        mag = torch.linalg.norm(x, ord=self.order, dim=-1)[..., None]   # [..., 1]
        return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))     # [..., 3]

    def forward(self, positions):
        return self.contract(positions)
