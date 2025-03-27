import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1",
              smooth=1, bin_size=1, **kwargs):
    """
    Plot data with optional smoothing and binning to show variability.

    :param data: list of DataFrames or a single DataFrame
    :param xaxis: column name for the x-axis
    :param value: column name for the y-axis
    :param condition: column name for different conditions/hue
    :param smooth: size of the smoothing window (moving average)
    :param bin_size: number of points to group together into a single bin
                     (e.g. 100 means each bin covers 100 steps)
    """

    # 1) Smooth the data if desired (applies a moving average).
    if smooth > 1:
        y = np.ones(smooth)
        for datum in data if isinstance(data, list) else [data]:
            arr = np.asarray(datum[value])
            z = np.ones(len(arr))
            smoothed_arr = np.convolve(arr, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_arr

    # 2) Concatenate list of DataFrames if necessary
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    # Ensure the condition column is categorical for nicer plotting
    data[condition] = data[condition].astype('category')

    # 3) Binning: group rows into bins of size `bin_size` along the x-axis
    #    - We create a new column "Bin" that identifies which bin a row belongs to.
    #    - If bin_size=1, no binning occurs (each row is its own bin).
    if bin_size > 1:
        data["Bin"] = data[xaxis] // bin_size
    else:
        data["Bin"] = data[xaxis]  # each row is its own bin

    # 4) Aggregate over each bin+condition combination:
    #    - We'll compute mean and std for `value`.
    grouped = data.groupby([condition, "Bin"], observed=False)[value].agg(["mean", "std"]).reset_index()

    # 5) Rename columns for clarity
    grouped.rename(columns={"mean": f"{value}_mean", "std": f"{value}_std"}, inplace=True)

    # 6) Plot using Seaborn/matplotlib:
    sns.set_theme(style="darkgrid", font_scale=1.5)

    # We’ll manually plot mean + shaded std region for each condition
    plt.figure(figsize=(8, 5))

    for cond in grouped[condition].unique():
        subdf = grouped[grouped[condition] == cond]
        xvals = subdf["Bin"] * bin_size  # or just subdf["Bin"] if you want bin indices
        yvals = subdf[f"{value}_mean"]
        ystds = subdf[f"{value}_std"]

        # Ensure all values are numeric
        xvals = np.array(xvals, dtype=np.float64)
        yvals = np.array(yvals, dtype=np.float64)
        ystds = np.array(ystds, dtype=np.float64)

        # Plot the mean line
        plt.plot(xvals, yvals, label=cond)

        # Plot the shaded area = mean ± std
        plt.fill_between(xvals, yvals - ystds, yvals + ystds, alpha=0.3)

    plt.xlabel(xaxis)
    plt.ylabel(value)
    plt.legend(loc='best').set_draggable(True)

    # Optional: if max x is large, use scientific notation
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)
    plt.show()
