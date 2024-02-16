import torch
import numpy as np


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