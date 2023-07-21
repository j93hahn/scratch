import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from typing import List


class ViolinPlot:
    def __init__(self, nrows=1, ncols=1, figsize=(10, 10)):
        self.ax: plt.Axes
        _, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    @staticmethod
    def adjacent_values(vals, q1, q3):
        """Compute whiskers. Values 1.5 times the IQR beyond the low and high quartiles are considered outliers."""
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    @staticmethod
    def set_axis_style(ax: plt.Axes, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)

    def compute_percentiles(self, vals: List[np.ndarray]):
        try:
            quartile1, medians, quartile3 = np.percentile(vals, [25, 50, 75], axis=1)
        except ValueError:  # vals' elements are not the same shape
            quartile1 = np.array([np.percentile(_ele, 25) for _ele in vals])
            medians = np.array([np.percentile(_ele, 50) for _ele in vals])
            quartile3 = np.array([np.percentile(_ele, 75) for _ele in vals])
        finally:
            whiskers = np.array([
                ViolinPlot.adjacent_values(sorted(array), q1, q3)
                for array, q1, q3 in zip(vals, quartile1, quartile3)
            ])
            whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
            return quartile1, medians, quartile3, whiskers_min, whiskers_max

    def plot(self,
        vals: List[np.ndarray],
        labels: List[str],
        xlabel: str,
        ylabel: str,
        title: str,
        save_path: str,
        showmeans=False,
        showmedians=False,
        showextrema=False
    ):
        assert len(vals) == len(labels), "vals and labels must have the same length"
        assert save_path.endswith(".png"), "save_path must end with .png"

        parts = self.ax.violinplot(vals, showmeans=showmeans, showmedians=showmedians, showextrema=showextrema)
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')     # set color
            pc.set_edgecolor('black')       # set edge color
            pc.set_alpha(1)                 # set opacity

        q1, med, q3, w_min, w_max = self.compute_percentiles(vals)
        inds = np.arange(1, len(med) + 1)
        self.ax.scatter(inds, med, marker='o', color='white', s=5, zorder=3)
        self.ax.vlines(inds, q1, q3, color='k', linestyle='-', lw=5)
        self.ax.vlines(inds, w_min, w_max, color='k', linestyle='-', lw=1)

        ViolinPlot.set_axis_style(self.ax, labels)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        plt.savefig(save_path, dpi=300)
        self._clear()

    def hue_plot(self, df: pd.DataFrame, x: str, y: str, hue: str, palette: str = "Set2"):
        """Plot a violin plot with hue, a.k.a. side-by-side violin plots. Unfinished function."""
        sns.violinplot(ax=self.ax, data=df, x=x, y=y, hue=hue, palette=palette)
        self.ax.set_title(f"{y} vs {x} with hue {hue}")
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        self.ax.legend(loc="upper right")
        self.ax.grid(True)
        plt.savefig(f"{y}_vs_{x}_with_hue_{hue}.png", dpi=300)
        self._clear()

    def _clear(self):
        """Clear the plot."""
        self.ax.clear()
