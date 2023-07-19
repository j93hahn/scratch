import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap


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
    Sinebow().register_mapping()
    data = np.load(Path(__file__).parent.parent.joinpath('garden_data.npy'))
    plt.imshow(data, cmap='Sinebow')
    plt.show()
    plt.close()
