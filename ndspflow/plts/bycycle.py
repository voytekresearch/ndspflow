"""FOOOF plotting functions for returning ready-to-embed html."""

import re
from itertools import cycle

import numpy as np
from scipy.stats import zscore

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bycycle.utils import get_extrema_df


def plot_bm(df_features, sig, fs, threshold_kwargs, xlim=None, plot_only_result=True):
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
        fig = make_subplots(rows=5, cols=1, vertical_spacing=0.01)

    # Plot normalized signal
    fig.add_trace(
        go.Scatter(x=times, y=sig, mode='lines', line=dict(color='black', width=2), name="Signal"),
        row=1, col=1
    )

    # Determine which samples are defined as bursting
    is_osc = np.zeros(len(sig), dtype=bool)
    df_burst = df_features[df_features['is_burst'].values]

    # Plot bursting signal
    n_bursts = len(df_burst)
    zeropad = 2 if n_bursts < 100 else 0
    zeropad = 3 if n_bursts < 1000 else 0
    zeropad = 4 if n_bursts >= 1000 else zeropad

    for idx, (_, cyc) in enumerate(df_burst.iterrows()):

        samp_start_burst = int(cyc['sample_last_' + side_e])
        samp_end_burst = int(cyc['sample_next_' + side_e] + 1)

        trace_name = 'Burst: {idx_fmt}'.format(idx_fmt=str(idx).zfill(zeropad))

        fig.add_trace(go.Scatter(x=times[samp_start_burst:samp_end_burst],
                                 y=sig[samp_start_burst:samp_end_burst], mode='lines',
                                 line=dict(color='red', width=2), name=trace_name), row=1, col=1)

    # Plot cycle points
    peaks = df_features['sample_' + center_e].values
    troughs = np.append(df_features['sample_last_' + side_e].values,
                        df_features['sample_next_' + side_e].values[-1])

    for color, points in zip(['rgb(191, 0, 191, 1)', 'rgb(0, 191, 191, 1)'], [peaks, troughs]):

        fig.add_trace(go.Scatter(x=times[points], y=sig[points], mode='markers',
                                 marker=dict(color=color, size=6)), row=1, col=1)

    if plot_only_result:

        fig.update_layout(
            autosize=False,
            width=1000,
            height=325,
            showlegend=False,
            yaxis_title="Voltage<br>(normalized)",
            xaxis_title="Time",
        )

    else:

        # Plot burst features
        burst_params = ['amp_fraction', 'amp_consistency', 'period_consistency', 'monotonicity',
                        'burst_fraction']
        burst_params = [param for param in burst_params if param in df_features.columns]

        ylabels = [param.replace("_", " ").capitalize() for param in burst_params]

        colors = cycle(['rgba(31, 119, 180, .2)', 'rgba(214, 39, 40, .4)', 'rgba(188, 189, 34, .4)',
                        'rgba(44, 160, 44, .2)', 'rgba(148, 103, 189, .4)', 'rgba(255, 127, 14, .4)'])

        for idx, burst_param in enumerate(burst_params):

            # Burst parameter
            fig.add_trace(
                go.Scatter(
                    x=times[df_features['sample_' + center_e]], y=df_features[burst_param],
                    mode='lines+markers', marker=dict(color='black')
                ),
                row=idx+2, col=1
            )

            # Horizontal threshold line
            thresh = threshold_kwargs[burst_param + '_threshold']
            fig.add_shape(type="line", x0=0, y0=thresh, x1=len(sig)/fs, y1=thresh,
                          line=dict(dash="dash"), row=idx+2, col=1)

            # Highlight sub-threshold regions
            rects_x = np.array([])
            fillcolor = next(colors)

            for _, cyc in df_features.iterrows():

                if cyc[burst_param] <= thresh:

                    last_side = times[cyc['sample_last_' + side_e]]
                    next_side = times[cyc['sample_next_' + side_e]]

                    rects_x = np.append(rects_x, [last_side, last_side, next_side, next_side])

            # Highlight signal
            smax = np.max(sig)
            smin = np.min(sig)

            sig_rects_y = np.repeat([[smax*2, smin*2, smin*2, smax*2]],
                                    len(df_features), axis=0).flatten()
            sig_rects_y = sig_rects_y + 0.1

            fig.add_trace(go.Scatter(x=rects_x, y=sig_rects_y, fill="toself",
                          fillcolor=fillcolor, mode="lines", line=dict(width=0)), row=1, col=1)

            # Highligh parameter
            param_rects_y = np.repeat([[0, 1.1, 1.1, 0]], len(df_features), axis=0).flatten()

            fig.add_trace(go.Scatter(x=rects_x, y=param_rects_y, fill="toself",
                          fillcolor=fillcolor, mode="lines", line=dict(width=0)), row=idx+2, col=1)

            # Axes settings
            ylabel = str(ylabels[idx] + "<br>threshold={thresh}").format(thresh=thresh)
            fig.update_yaxes(title_text=ylabel, range=[0, 1.1], dtick=.2, row=idx+2, col=1)
            fig.update_yaxes(range=[smin+.1, smax+.1], row=1, col=1)

            if idx == len(burst_params)-1:
                fig.update_xaxes(title_text='Time', showticklabels=True, row=idx+2, col=1)
            else:
                fig.update_xaxes(showticklabels=False, row=idx+2, col=1)


        # Add time label to last subplot
        fig.update_xaxes(title_text='Time', showticklabels=True, row=len(burst_params)+2, col=1)

        # Update signal axes
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="Voltage<br>(normalized)", row=1, col=1)

        # Zoom link across all subplots
        fig.update_xaxes(matches="x")

        # Update layout across all subplots
        fig.update_layout(
            autosize=False,
            width=1000,
            height=1200,
            showlegend=False
        )

    graph = fig.to_html()

    return graph
