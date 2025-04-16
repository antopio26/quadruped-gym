# src/utils/plot.py
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

# Modified function signature to accept title and output explicitly
def plot_data_line(data, xaxis='Epoch', value="AverageEpRet", std="Std",
                   condition="Condition1", smooth=1, title=None, output=None, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

            if std in datum:
                std_x = np.asarray(datum[std])
                smoothed_std = np.convolve(std_x, y, 'same') / np.convolve(z, y, 'same')
                datum[std] = smoothed_std

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set_theme(style="darkgrid", font_scale=1.5)

    # Pass only the remaining relevant kwargs to seaborn
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, **kwargs)

    # plot standard deviation as filled region if 'Std' column exists
    if std in data.columns:
        # Ensure data is sorted by xaxis for correct fill_between
        data_sorted = data.sort_values(by=xaxis)
        plt.fill_between(data_sorted[xaxis], data_sorted[value] - data_sorted[std], data_sorted[value] + data_sorted[std], alpha=0.3)
    elif 'Std' in data.columns: # Check the original std name too
         data_sorted = data.sort_values(by=xaxis)
         plt.fill_between(data_sorted[xaxis], data_sorted[value] - data_sorted['Std'], data_sorted[value] + data_sorted['Std'], alpha=0.3)


    plt.legend(loc='best').set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Set the title if provided
    if title:
        plt.title(title)

    plt.tight_layout(pad=0.5)

    # Save the plot if output path is provided
    if output:
        plt.savefig(output)
        print(f"Plot saved to {output}") # Optional: confirmation message
    # else:
        # plt.show() # Or show the plot if not saving

# --- rest of the file remains the same ---
# ... (plot_data and plot_reward_components functions) ...

def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1",
              smooth=1, bin_size=1, title=None, output=None, **kwargs): # Added title/output here too for consistency
    """
    Plot data with optional smoothing and binning to show variability.

    :param data: list of DataFrames or a single DataFrame
    :param xaxis: column name for the x-axis
    :param value: column name for the y-axis
    :param condition: column name for different conditions/hue
    :param smooth: size of the smoothing window (moving average)
    :param bin_size: number of points to group together into a single bin
                     (e.g. 100 means each bin covers 100 steps)
    :param title: Optional title for the plot
    :param output: Optional path to save the plot image
    """

    # 1) Smooth the data if desired (applies a moving average).
    if smooth > 1:
        y = np.ones(smooth)
        # Handle both list of dataframes and single dataframe
        data_list = data if isinstance(data, list) else [data]
        processed_data = []
        for datum in data_list:
            datum_copy = datum.copy() # Avoid modifying original dataframe
            arr = np.asarray(datum_copy[value])
            z = np.ones(len(arr))
            smoothed_arr = np.convolve(arr, y, 'same') / np.convolve(z, y, 'same')
            datum_copy[value] = smoothed_arr
            processed_data.append(datum_copy)
        # Reconstruct original structure
        data = processed_data if isinstance(data, list) else processed_data[0]


    # 2) Concatenate list of DataFrames if necessary
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    # Ensure the condition column is categorical for nicer plotting
    data[condition] = data[condition].astype('category')

    # 3) Binning: group rows into bins of size `bin_size` along the x-axis
    if bin_size > 1:
        # Ensure xaxis is numeric before binning
        data[xaxis] = pd.to_numeric(data[xaxis], errors='coerce')
        data = data.dropna(subset=[xaxis]) # Drop rows where xaxis couldn't be converted
        data["Bin"] = (data[xaxis] // bin_size).astype(int)
        # Use the mean x-value within the bin for plotting
        bin_centers = data.groupby([condition, "Bin"], observed=False)[xaxis].mean().reset_index()
        bin_centers.rename(columns={xaxis: f"{xaxis}_mean"}, inplace=True)
    else:
        data["Bin"] = data[xaxis]  # each row is its own bin
        bin_centers = data[[condition, "Bin", xaxis]].rename(columns={xaxis: f"{xaxis}_mean"})


    # 4) Aggregate over each bin+condition combination:
    grouped = data.groupby([condition, "Bin"], observed=False)[value].agg(["mean", "std"]).reset_index()

    # Merge bin centers back for plotting
    grouped = pd.merge(grouped, bin_centers, on=[condition, "Bin"])


    # 5) Rename columns for clarity
    grouped.rename(columns={"mean": f"{value}_mean", "std": f"{value}_std"}, inplace=True)

    # 6) Plot using Seaborn/matplotlib:
    sns.set_theme(style="darkgrid", font_scale=1.5)

    plt.figure(figsize=(8, 5)) # Consider making figsize configurable

    for cond in grouped[condition].unique():
        subdf = grouped[grouped[condition] == cond].sort_values(by=f"{xaxis}_mean") # Sort for plotting
        xvals = subdf[f"{xaxis}_mean"] # Use mean x-value of the bin
        yvals = subdf[f"{value}_mean"]
        ystds = subdf[f"{value}_std"].fillna(0) # Replace NaN std with 0 for fill_between

        # Ensure all values are numeric
        xvals = np.array(xvals, dtype=np.float64)
        yvals = np.array(yvals, dtype=np.float64)
        ystds = np.array(ystds, dtype=np.float64)

        # Plot the mean line
        plt.plot(xvals, yvals, label=cond, **kwargs) # Pass remaining kwargs here

        # Plot the shaded area = mean ± std
        plt.fill_between(xvals, yvals - ystds, yvals + ystds, alpha=0.3)

    plt.xlabel(xaxis)
    plt.ylabel(value)
    plt.legend(loc='best').set_draggable(True)

    # Set title if provided
    if title:
        plt.title(title)

    # Optional: if max x is large, use scientific notation
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)

    # Save or show the plot
    if output:
        plt.savefig(output)
        print(f"Plot saved to {output}")
    # else:
    #     plt.show()


def plot_reward_components(data, output, exclude_cols=['Condition', 'Std', 'Training Steps', 'alive_bonus']):
    # Ensure 'Training Steps' exists
    if 'Training Steps' not in data.columns:
        print("Warning: 'Training Steps' column not found in data for plot_reward_components.")
        # Option 1: Try to find another suitable x-axis or return
        # Option 2: Use index if appropriate (less ideal)
        # data['Training Steps'] = data.index
        return # Exit if no suitable x-axis

    columns_to_plot = [col for col in data.columns if col not in exclude_cols and col != 'Reward'] # Exclude total Reward too

    fig = go.Figure()

    # Add total reward separately for emphasis maybe?
    if 'Reward' in data.columns:
         fig.add_trace(go.Scatter(
             x=data['Training Steps'],
             y=data['Reward'],
             mode='lines',
             name='Total Reward', # Explicit name
             line=dict(width=3, color='black') # Make it stand out
         ))


    for col in columns_to_plot:
        # Ensure column exists before plotting
        if col in data.columns:
            fig.add_trace(go.Scatter(
                x=data['Training Steps'],
                y=data[col],
                mode='lines',
                name=col
            ))
        else:
            print(f"Warning: Column '{col}' not found in data for plot_reward_components.")


    fig.update_layout(
        title='Reward Components and Total Reward Over Training', # More descriptive title
        xaxis_title='Training Steps',
        yaxis_title='Value', # More general y-axis title
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,   # Adjust position slightly above plot area
            xanchor='left', # Align legend to the left
            x=0
        ),
        margin=dict(t=80, b=40, l=50, r=30), # Adjust top margin for title/legend
        height=600, # Consider making height configurable
        template='plotly_white' # Or other templates like 'plotly_dark'
    )

    try:
        fig.write_html(output)
    except Exception as e:
        print(f"Error saving plotly figure to {output}: {e}")

