import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import *

seaborn_white = lambda: plt.style.use('seaborn-v0_8-whitegrid')


def trailing_window_view(xs, window_size):
    assert (window_size % 2) == 1, "window size should be odd"
    view = np.lib.stride_tricks.sliding_window_view(
        np.pad(xs, (window_size - 1, 0), mode="edge"), window_size
    )
    return view


def make_plot(ax, xs, psnrs, label, ws=51):
    data = trailing_window_view(psnrs, ws)
    μ = data.mean(-1)
    σ = data.std(-1)
    ax.plot(xs, μ, label=label, alpha=0.7)
    ax.fill_between(xs, μ - σ, μ + σ, alpha=0.3)


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

    h = grid[0].imshow(vals.reshape(dims[0], dims[1]), cmap=cmap, norm=norm(lower, upper))
    grid[0].set_title(f'Visualizing {name.capitalize()}: Iteration {iteration}')
    grid[0].get_xaxis().set_visible(False)
    grid[0].get_yaxis().set_visible(False)

    plt.colorbar(h, cax=grid.cbar_axes[0])
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_iteration_{iteration}.png', dpi=300)
    plt.close()


def sinebow():
    ...
