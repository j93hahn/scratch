import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from matplotlib.colors import *


"""
Visualize an array as a 2D image with a colormap and normalized colorbar.
"""
def visualize_norm_cmap_2d(
    vals: np.ndarray,           # values to visualize
    name: str,                  # name of the visualization
    iteration: int,             # iteration number
    figsize: tuple=(5, 5),      # figure size
    dims: tuple=(800, 800),     # image dimensions
    cmap: str='viridis',        # colormap to use
    lower: float=1e-2,          # lower bound of the colormap
    upper: float=1e2,           # upper bound of the colormap
    norm: str='log',            # normalization to use
):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(
        fig, 111, nrows_ncols=(1, 1),
        cbar_location="right", cbar_mode="edge", cbar_size="7%", cbar_pad=0.15,
    )

    vals[vals < lower] = lower
    vals[vals > upper] = upper

    norm = {
        'log': LogNorm,
        'linear': Normalize,
        'symlog': SymLogNorm,
        'power': PowerNorm,
        'asinh': AsinhNorm,
    }[norm]

    g: Axes = grid[0]
    img = g.imshow(vals.reshape(dims[0], dims[1]), cmap=cmap, norm=norm(lower, upper))
    g.set_title(f'Visualizing {name.capitalize()}: Iteration {iteration}')
    g.get_xaxis().set_visible(False)
    g.get_yaxis().set_visible(False)

    plt.colorbar(img, cax=grid.cbar_axes[0])
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_iteration_{iteration}.png', dpi=300)
    plt.close()
