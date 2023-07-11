import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
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


"""
Cyclic and uniform colormap. Taken from http://basecase.org/env/on-rainbows.
"""
class Sinebow():
    def __init__(self):
        self.registered = False

    @staticmethod
    def rgb2hex(rgb):
        return '#%02x%02x%02x' % tuple(rgb)

    def sinebow(self, n):

        def compute_color(h):
            h = (h + 0.5) * -1
            c = torch.stack([
                torch.sin(torch.pi * h),
                torch.sin(torch.pi * (h + 1/3)),
                torch.sin(torch.pi * (h + 2/3)),
            ], axis=-1)
            return (255 * c ** 2).type(torch.int16)

        phi = (1+5**0.5)/2
        return compute_color(n * phi)

    def register_mapping(self, n=130, seed=1451):
        """
        Register the Sinebow mapping with matplotlib. Accessible via cmap='Sinebow'. We
            generate a random n x 3 array of RGB values and register it as a colormap. I've
            found that n=130 is a good value for the number of sample colors to use. The
            seed is set for reproducibility; its value impacts the colors generated.

        Note: plt.register_cmap('Sinebow', sinebow_cmap) is deprecated. Use
            mpl.colormaps.register(sinebow_cmap) instead.
        """
        if self.registered:
            return
        self.registered = True
        torch.manual_seed(seed=seed) # for reproducibility
        sinebow_cmap = ListedColormap(self.sinebow(torch.randn((n))).numpy() / 256)
        mpl.colormaps.register(sinebow_cmap, name='Sinebow')


if __name__ == '__main__':
    x = Sinebow()
    x.register_mapping()

    data = np.load('garden_data.npy')
    plt.imshow(data, cmap='Sinebow')
    plt.show()
    plt.close()
