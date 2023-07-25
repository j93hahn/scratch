import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.collections import PolyCollection
from typing import List, Optional, Any
from torch import Tensor


class ViolinPlot:
    def __init__(self, nrows=1, ncols=1, figsize=(10, 10)):
        self.ax: plt.Axes
        _, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    def plot(
        self, vals: List[np.ndarray], labels: List[str], xlabel: str,
        ylabel: str, title: str, save_path: str, showmeans=False,
        showmedians=False, showextrema=False
    ):
        """Plot a violin plot without hue. Generates vertical violin plots only. Deprecated:
            use plot_v2() instead. Matplotlib's API lacks the versability and customizability
            of seaborn's API."""
        assert len(vals) == len(labels), "vals and labels must have the same length"
        assert save_path.endswith(".png"), "save_path must end with .png"

        parts = self.ax.violinplot(vals, showmeans=showmeans, showmedians=showmedians, showextrema=showextrema)
        pc: PolyCollection
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')     # set color
            pc.set_edgecolor('black')       # set edge color
            pc.set_alpha(1)                 # set opacity

        q1, med, q3, w_min, w_max = self.compute_percentiles(vals)
        inds = np.arange(1, len(med) + 1)
        self.ax.scatter(inds, med, marker='o', color='white', s=5, zorder=3)
        self.ax.vlines(inds, q1, q3, color='k', linestyle='-', lw=5)
        self.ax.vlines(inds, w_min, w_max, color='k', linestyle='-', lw=1)

        self.set_x_axis_style(labels)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        plt.savefig(save_path, dpi=300)
        self._clear()

    def plot_v2(
        self, labels: List[str], data: List[Any], x: str, y: str, title: str,
        xlabel: str, ylabel: str, path: str, palette: str="Set2"
    ):
        """Plot a violin plot without hue. Generates vertical violin plots only. Mirror
            implementation of plot() above, but using seaborn's API."""
        df = self._extract_pandas_df(split=False, labels=labels, data=data)
        assert x in labels and y in labels
        sns.violinplot(data=df, x=x, y=y, palette=palette, ax=self.ax)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        plt.savefig(path, dpi=300)
        self._clear()

    def split_plot(
        self, labels: List[str], data: List[Any], data2: List[Any],
        x: str, y: str, hue: str, title: str, xlabel: str, ylabel: str,
        path: str, palette: str="Set2", legend_loc: str="upper right"
    ):
        """Plot a violin plot with hue, a.k.a. side-by-side split violin plots. Generates
            vertical violin plots only.

        Args:
            labels (List[str]): the labels for the data
            data (List[Any]): the data to plot
            data2 (List[Any]): the second data to plot
            x (str): the variable where we observe the distributions
            y (str): the variable whose distribution is to be visualized
            hue (str): the variable to split the data by
            title (str): the title of the plot
            xlabel (str): the x-axis label
            ylabel (str): the y-axis label
            path (str): the path to save the plot
            palette (str, optional): the color palette. Defaults to 'Set2'.
            legend_loc (str, optional): the location of the legend. Defaults to 'upper right'.

        Returns:
            None. Produces a split violin plot along the specified orientation.
        """
        df = self._extract_pandas_df(split=True, labels=labels, data=data, data2=data2)
        assert x in labels and y in labels and hue in labels
        sns.violinplot(data=df, x=x, y=y, hue=hue, split=True, palette=palette, ax=self.ax)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend(loc=legend_loc)
        plt.savefig(path, dpi=300)
        self._clear()

    @staticmethod
    def adjacent_values(vals, q1, q3):
        """Compute whiskers. Values 1.5 times the IQR beyond the low and high quartiles are considered outliers."""
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_x_axis_style(self, labels):
        self.ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        self.ax.set_xlim(0.25, len(labels) + 0.75)

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

    def _extract_pandas_df(self, split: bool, labels: List[str], data: List[Any], data2: Optional[List[Any]] = None) -> pd.DataFrame:
        """Extract a pandas DataFrame from the data and labels."""
        assert len(labels) == len(data), f"labels and data must have the same length, got {len(labels)} and {len(data)}"
        df = pd.DataFrame(dict(zip(labels, data)))
        if split:   # concatenate data2 to df; the hue variable will split the data into two groups
            assert data2 is not None, "data2 must be provided when split is True"
            assert len(data) == len(data2), f"data and data2 must have the same length, got {len(data)} and {len(data2)}"
            _df2 = pd.DataFrame(dict(zip(labels, data2)))
            df = pd.concat([df, _df2])

        for i in range(len(labels)):    # ensure each cell contains one piece of information
            df = df.explode(labels[i])
            if isinstance(data[i], list) and (isinstance(data[i][0], float) or \
                isinstance(data[i][0], np.ndarray) or isinstance(data[i][0], Tensor)):
                # convert dtype to float from object if list of floats, np.ndarray, or torch.tensor
                df[labels[i]] = df[labels[i]].astype(float)

        return df

    def _clear(self):
        """Clear the plot."""
        plt.cla()

    def __del__(self):
        """Close the plot."""
        plt.close()


if __name__ == '__main__':
    x = ViolinPlot()
    x.split_plot(['layer', 'distribution', 'label'], [[0, 1, 2, 3, 4], [np.random.randn(50) for _ in range(5)], 'coarse'], \
                [[0, 1, 2, 3, 4], [np.random.randn(50) for _ in range(5)], 'fine'], 'layer', 'distribution', 'label', \
                'Visualizing Activation Statistics by Layer', 'Layer', 'Activation Distribution', 'split_violin.png')
    x.plot_v2(['layer', 'distribution'], [[0, 1, 2, 3, 4], [np.random.randn(50) for _ in range(5)]], 'layer', 'distribution', \
            'Visualizing Activation Statistics by Layer', 'Layer', 'Activation Distribution', 'violin.png')
