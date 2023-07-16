import torch
from torchtyping import TensorType


def sample_grid_box(gridSize: TensorType["x y z"], device: torch.device):
    gridSamples = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.5, 1.5, gridSize[0]),
            torch.linspace(-1.5, 1.5, gridSize[1]),
            torch.linspace(-1.5, 1.5, gridSize[2]),
        ), -1
    ).reshape(-1, 3).to(device) # [N_samples, 3]
    return gridSamples


def sample_grid_sphere(gridSize: TensorType["s s s"], device: torch.device):
    gridSamples = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.5, 1.5, gridSize[0]),
            torch.linspace(-1.5, 1.5, gridSize[1]),
            torch.linspace(-1.5, 1.5, gridSize[2]),
        ), -1
    ).reshape(-1, 3).to(device)
    gridSamples = gridSamples[torch.norm(gridSamples, dim=-1) < 1.5]
    return gridSamples


if __name__ == '__main__':
    x = sample_grid_sphere(torch.tensor([100, 100, 100]), torch.device("cpu"))
