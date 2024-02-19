import torch
import torch.nn as nn


class HashEncoder(nn.Module):
    """
    personal PyTorch re-implementation of instant-ngp hash grid. based off in part by
    Zip-NeRF's JAX re-implementation https://github.com/jonbarron/camp_zipnerf/tree/main
    """
    def __init__(self):
        super().__init__()
