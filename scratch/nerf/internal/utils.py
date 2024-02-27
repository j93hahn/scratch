import numpy as np
import torch
import torch.nn as nn


def fourier_features(x, B):
    """
    applies fourier feature mapping to the input x; B is sampled from a Gaussian distribution

    input arguments:
        - x has shape [..., d] where d is the number of features
        - B has shape [m, d] where m is the number of fourier features
            - assume B is sampled from a Gaussian distribution with 0 mean and variance sigma^2
                where sigma is a hyperparameter

    returns:
        x_proj has shape [..., 2m]
    """
    x_proj = x @ B.T
    x_proj = 2 * np.pi * x_proj
    x_proj = torch.cat(
        [torch.cos(x_proj), torch.sin(x_proj)],
        dim=-1
    )
    assert x_proj.shape[-1] == 2 * B.shape[0] and x_proj.shape[:-1] == x.shape[:-1], \
        f"Expected shape [..., {2 * B.shape[0]}], got {x_proj.shape}"
    return x_proj


def get_final_image(weights, quantity):
    """
    a general purpose function to compute the final image from the weights and the quantity

    supports rgb, density, and normals
    """
    image = torch.sum(weights * quantity, dim=-1)
    return image


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


class SceneBox(nn.Module):
    def __init__(self, aabb):
        super().__init__()
        self.aabb = aabb

    def forward(self, positions):
        raise NotImplementedError("SceneBox is not implemented yet")
        return torch.where(
            (positions > self.aabb[0]) & (positions < self.aabb[1]),
            positions,
            torch.zeros_like(positions)
        )
