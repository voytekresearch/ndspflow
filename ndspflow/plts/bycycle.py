"""FOOOF plotting functions for returning ready-to-embed html."""

import re
import numpy as np
from scipy.stats import truncnorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import zscore
from bycycle.utils import limit_df, limit_signal, get_extrema_df



def plot_bycycle(df_features, sig, fs, threshold_kwargs, xlim=None, plot_only_result=True):
    """Plot a individual bycycle fits.

    Parameters
    ----------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle.
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    threshold_kwargs : dict, optional, default: None
        Feature thresholds for cycles to be considered bursts.
    xlim : tuple of (float, float), optional, default: None
        Start and stop times for plot.
    plot_only_result : bool, optional, default: True
        Plot only the signal and bursts, excluding burst parameter plots.

    Returns
    -------
    graph : str
        The fooof plot as a string containing html.
    """

    # Normalize signal
    sig = zscore(sig)

    # Determine time array and limits
    times = np.arange(0, len(sig) / fs, 1 / fs)
    xlim = (times[0], times[-1]) if xlim is None else xlim

    # Determine if peak of troughs are the sides of an oscillation
    center_e, side_e = get_extrema_df(df_features)

    # Remove this kwarg since it isn't stored cycle by cycle in the df (nothing to plot)
    if 'min_n_cycles' in threshold_kwargs.keys():
        del threshold_kwargs['min_n_cycles']

    n_kwargs = len(threshold_kwargs.keys())

    # Create figure and subplots
    if plot_only_result:
        fig = make_subplots(rows=1, cols=1)
    else:
        fig = make_subplots(rows=5, cols=1)

    fig.add_trace(go.Scatter(x=times, y=sig, mode='lines',  line=dict(color='black', width=2)),
                  row=1, col=1)

    # Determine which samples are defined as bursting
    is_osc = np.zeros(len(sig), dtype=bool)
    df_burst = df_features[df_features['is_burst'].values]

    # Plot non-burst signal
    for _, cyc in df_burst.iterrows():

        samp_start = int(cyc['sample_last_' + side_e])
        samp_end = int(cyc['sample_next_' + side_e] + 1)

        fig.add_trace(go.Scatter(x=times[samp_start:samp_end],
                                 y=sig[samp_start:samp_end], mode='lines',
                                 line=dict(color='red', width=2)), row=1, col=1)

    # Plot bursting signal
    for _, cyc in df_burst.iterrows():

        samp_start_burst = int(cyc['sample_last_' + side_e])
        samp_end_burst = int(cyc['sample_next_' + side_e] + 1)

        fig.add_trace(go.Scatter(x=times[samp_start_burst:samp_end_burst],
                                 y=sig[samp_start_burst:samp_end_burst], mode='lines',
                                 line=dict(color='red', width=2)), row=1, col=1)

    # Plot cycle points
    peaks = df_features['sample_' + center_e].values
    troughs = np.append(df_features['sample_last_' + side_e].values,
                        df_features['sample_next_' + side_e].values[-1])

    for color, points in zip(['rgb(191, 0, 191, 1)', 'rgb(0, 191, 191, 1)'], [peaks, troughs]):

        fig.add_trace(go.Scatter(x=times[points], y=sig[points], mode='markers',
                                 marker=dict(color=color, size=6)), row=1, col=1)

    fig.update_layout(
        autosize=False,
        width=1000,
        height=300,
        showlegend=False
    )

    graph = fig.to_html()

    return graph
