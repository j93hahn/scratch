import matplotlib.pyplot as plt
import numpy as np


seaborn_white = lambda: plt.style.use('seaborn-v0_8-whitegrid')

def trailing_window_view(xs, window_size):
    """Generate a trailing window view of the input array."""
    assert (window_size % 2) == 1, "window size should be odd"
    view = np.lib.stride_tricks.sliding_window_view(
        np.pad(xs, (window_size - 1, 0), mode="edge"), window_size
    )
    return view


def make_plot(ax: plt.Axes, xs, ys, label, ws=51):
    """Generate a simple line plot with a shaded region indicating the standard deviation."""
    data = trailing_window_view(ys, ws)
    μ = data.mean(-1)
    σ = data.std(-1)
    ax.plot(xs, μ, label=label, alpha=0.7)
    ax.fill_between(xs, μ - σ, μ + σ, alpha=0.3)
