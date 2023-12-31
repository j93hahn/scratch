import torch


def sample_grid_box(gridSize=200, aabb=[-1.5, 1.5], device='cuda'):
    gridSamples = torch.stack(
        torch.meshgrid(
            torch.linspace(aabb[0], aabb[1], gridSize),
            torch.linspace(aabb[0], aabb[1], gridSize),
            torch.linspace(aabb[0], aabb[1], gridSize),
        ), -1
    ).reshape(-1, 3).to(device) # [N_samples, 3]
    return gridSamples
