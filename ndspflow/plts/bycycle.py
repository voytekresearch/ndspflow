"""Bycycle plotting functions for returning ready-to-embed html."""

from itertools import cycle
import re
from os import path

import numpy as np
from scipy.stats import zscore
from scipy.signal import resample
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bycycle.utils import get_extrema_df

from ndspflow.core.utils import flatten_bms


def plot_bm(df_features, sig, fs, threshold_kwargs, df_idx, xlim=None, plot_only_result=True):
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
    df_idx : int
        The index of the dataframe in the javascript array. This is only used for fetching data
        for the relabel js callback.
    xlim : tuple of (float, float), optional, default: None
        Start and stop times for plot.
    plot_only_result : bool, optional, default: True
        Plot only the signal and bursts, excluding burst parameter plots.

    Notes
    -----
    The output_dir and df_idx arguments are used to fetch a javascript array containing the
    dataframe results during the js relabel callback. This is done since javascript can't access
    the local filesystem to load csv files.

    Returns
    -------
    graph : str
        The bycycle plot as a string containing html.
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

    # Create figure and subplots
    if plot_only_result:
        fig = go.Figure(make_subplots(rows=1, cols=1))
    elif not plot_only_result:
        fig = go.Figure(make_subplots(rows=5, cols=1, vertical_spacing=0.01))

    # Plot bursts
    fig = _plot_bursts(df_features, sig, times, center_e, side_e, fig, row=1, col=1)

    # Plot params
    if not plot_only_result:

        burst_params = ['amp_fraction', 'amp_consistency', 'period_consistency', 'monotonicity',
                        'burst_fraction']

        burst_params = [param for param in burst_params if param in df_features.columns]

        fig = _plot_params(df_features, sig, fs, times, center_e, side_e,
                           burst_params, threshold_kwargs, fig, row=2, col=1)

    # Update axes and layout
    if not plot_only_result:

        # Add time label to last subplot
        fig.update_xaxes(title_text='Time', showticklabels=True, row=len(burst_params)+2, col=1)

        # Update signal axes
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="Voltage<br>(normalized)", row=1, col=1)

        # Zoom link across all subplots
        fig.update_xaxes(matches="x")

        # Update layout across all subplots
        fig.update_layout(width=1000, height=1000)

    else:

        fig.update_layout(width=1000, height=325, xaxis_title="Time",
                          yaxis_title="Voltage<br>(normalized)")

    fig.update_layout(
        autosize=True,
        showlegend=False,
        title_text="Burst Detection Plots",
    )

    fig.update_xaxes(rangeslider= {'visible':True}, row=5, col=1)

    graph = fig.to_html(include_plotlyjs=False, full_html=False)

    # Get burst traces
    trace_id = len(fig.data) - 2
    burst_traces = [idx for idx in range(0, len(df_features))]

    # Get the div id containg the plot
    div_id = re.search("<div id=.+?\"", graph)[0]
    div_id = div_id[9:-1]

    # Create js callback
    js_callback = ["<script type=\"text/javascript\">\n"]
    js_callback.append("""
    var burstPlot = document.getElementById('{plot_id}');
    var burstTraces = {burst_traces};
    var traceId = {trace_id};
    burstPlot.on('plotly_click', function(data){{
        relabel1DBursts(data, burstPlot, {idx}, burstTraces, traceId);
    }});
    """.format(trace_id=trace_id, burst_traces=str(burst_traces), plot_id=div_id, idx=df_idx)
    )
    js_callback.append("</script>\n")
    js_callback = "".join(js_callback)

    # Flatten lists into single string
    graph = re.sub("</body>\n</html>", "\n", graph)

    # Add recompute burst btn below the plots
    rewrite_call = "rewriteBursts({div_id})".format(div_id=str([div_id]))
    btn = "\n\t\t<p><center><button onclick=\"" + rewrite_call + "\" class=\"btn\" "
    btn = btn + "title=\"update is_burst column\">Update Bursts</button></center></p>"

    graph = graph + js_callback + btn + "\n</body>\n</html>"

    return graph


def plot_bg(dfs_features, sigs, fs, titles=None, btn=True, xlim=None):
    """Plot 2D bycycle results.

    Parameters
    ----------
    dfs_features : list of pandas.DataFrame
        Dataframes containing shape and burst features for each cycle.
    sigs : 2d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    titles : list, optional, default: None
        The titles for each subplot.
    btn : bool, optional, default: True
        Adds a recompute bursts button when True. Omits when False.
    xlim : tuple of (float, float), optional, default: None
        Start and stop times for plot.

    Returns
    -------
    graph : str
        The bycycle plot as a string containing html.
    """

    # Initialize figures in groups of 10
    #   Plotly doesn't render single figures well with 100+ plots
    n_per_fig = 10
    n_figs = int(np.ceil(len(sigs) / n_per_fig))
    figs = np.zeros(n_figs).tolist()

    titles = [] if titles is None else titles
    n_rows = []
    for idx in range(n_figs):

        # Subplot titles
        start = idx * n_per_fig
        end = start + len(sigs[start:start+n_per_fig])

        if len(titles) == idx:
            titles.append("Indices: {start}-{end}".format(start=start, end=end))

        # Create subplots
        n_rows.append(end - start)
        fig = make_subplots(rows=n_rows[idx], cols=1, vertical_spacing=0.005, shared_xaxes=True)
        figs[idx] = fig

    for idx, df_features in enumerate(dfs_features):

        # Normalize signal
        sig = zscore(sigs[idx])

        # Determine time array and limits
        times = np.arange(0, len(sig) / fs, 1 / fs)
        xlim = (times[0], times[-1]) if xlim is None else xlim

        # Get extrema
        center_e, side_e = get_extrema_df(df_features)

        # Plot bursts
        fig_idx = int(np.ceil((idx+1) / n_per_fig)) - 1
        row_idx = int(idx - (n_per_fig * fig_idx))

        _plot_bursts(df_features, sig, times, center_e, side_e,
                     figs[fig_idx], plot_cps=False, row=row_idx+1, col=1)

    graphs = []
    for idx, fig in enumerate(figs):

        # The size of plotly subplots don't scale properly, this is a workaround
        height = (60 * n_rows[idx]) + ((n_per_fig - n_rows[idx]) * 18)

        # Update the figures
        figs[idx].update_layout(
            autosize=False,
            width=1000,
            height=height,
            showlegend=False,
            margin_autoexpand=False,
            title_text=titles[idx]
        )
        figs[idx].update_yaxes(showticklabels=False, showgrid=False)
        figs[idx].update_xaxes(showgrid=False)
        figs[idx].update_xaxes(rangeslider= {'visible':True}, row=n_rows[idx], col=1)

        # Convert to html
        if idx == 0:

            graphs.append(fig.to_html(include_plotlyjs=False, full_html=False))

        else:

            div = re.search("<div>.*</div>", fig.to_html(include_plotlyjs=False,
                                                         full_html=False))[0]
            graphs.append(div + "\n")

    # Custom js callback
    js_callback = ["<script type=\"text/javascript\">\n"]

    # Get plot div ids
    div_ids = [re.search("<div id=.+?\"", graph)[0][9:-1] for graph in graphs]

    # Recolor (non)bursts on click
    for div_id in div_ids:

        js_callback.append(
        """
        recolorBursts('{plotID}');
        """.format(plotID=div_id)
        )

    js_callback.append("</script>\n")

    # Flatten lists into single string
    graphs.append("".join(js_callback))

    if btn:

        rewrite_call = "rewriteBursts({div_ids})".format(div_ids=div_ids)

        # Add a button
        btn = "\n\t\t<p><center><button onclick=\"" + rewrite_call + "\" class=\"btn\" "
        btn = btn + "title=\"update is_burst column\">Update Bursts</button></center></p>"
        graphs.append(btn)

    graphs[0] = re.sub("</body>\n</html>", "", graphs[0])
    graphs = "".join(graphs)

    return graphs


def plot_bgs(dfs_features, sigs, fs, xlim=None):
    """Plot 3D bycycle results.

    Parameters
    ----------
    dfs_features : list of pandas.DataFrame
        Dataframes containing shape and burst features for each cycle.
    sigs : 2d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    xlim : tuple of (float, float), optional, default: None
        Start and stop times for plot.

    Returns
    -------
    graph : str
        The bycycle plot as a string containing html.
    """

    # Number of plots per figure
    n_per_fig = 10
    dfs_features_2d, _, sigs_2d = flatten_bms(dfs_features, '', sigs)

    # Create subplot titles
    dim0 = np.shape(sigs)[0]
    dim1 = np.shape(sigs)[1]

    starts = [(i, j) for i in range(dim0) for j in range(dim1)][::n_per_fig]
    ends = starts[1:]
    ends.append((dim0-1, dim1-1))

    titles = []
    for start, end in zip(starts, ends):

        titles.append("Indices: [{s0}][{s1}] - [{e0}][{e1}]".format(s0=start[0], s1=start[1],
                                                                    e0=end[0], e1=end[1]))

    graphs = plot_bg(dfs_features_2d, sigs_2d, fs, titles=titles, xlim=None)

    return graphs


def _plot_bursts(df_features, sig, times, center_e, side_e, fig,
                 plot_cps=True, row=1, col=1, dtype="float", ds_factor='adaptive'):
    """Plot where a signal is bursting"""

    # Downsample signal
    if ds_factor == 'adaptive':
        ds_factor = 1 if len(sig) < 1000 else 750 / len(sig)

    n_samples = round(len(sig) * ds_factor)
    sig_ds, times_ds = resample(sig, n_samples, t=times)

    sig_ds = sig
    times_ds = times

    # Plot cycle-by-cycle
    for _, cyc in df_features.iterrows():

        samp_end = int(cyc['sample_next_' + side_e])
        samp_start = int(cyc['sample_last_' + side_e])

        if ds_factor != 1:

            samp_start = np.argmin(np.abs(times_ds-times[samp_start]))
            samp_end = np.argmin(np.abs(times_ds-times[samp_end]))


        times_cyc = times_ds[samp_start:samp_end]
        sig_cyc = sig_ds[samp_start:samp_end]

        # Plotting is slightly faster with an int array
        if dtype == "int":
            sig_cyc = sig_cyc * 100
            sig_cyc = sig_cyc.astype(int)

        # Plot cycle
        if cyc['is_burst']:
            fig.add_trace(
                go.Scattergl(x=times_cyc, y=sig_cyc, mode='lines', name="Burst",
                             line=dict(color='red', width=2)),
                row=row, col=col
            )

        else:

            fig.add_trace(
                go.Scattergl(x=times_cyc, y=sig_cyc, mode='lines', name="Signal",
                             line=dict(color='black', width=2)),
                row=row, col=col
            )

    if plot_cps:
        # Centers
        centers = df_features['sample_' + center_e].values
        fig.add_trace(go.Scattergl(x=times[centers], y=sig[centers], mode='markers',
                                   name=str(center_e.capitalize()),
                                   marker=dict(color='rgb(191, 0, 191)', size=6)),
                      row=row, col=col)

        # Sides
        sides = np.append(df_features['sample_last_' + side_e].values,
                        df_features['sample_next_' + side_e].values[-1])
        fig.add_trace(go.Scattergl(x=times[sides], y=sig[sides], mode='markers',
                                   name=str(side_e.capitalize()),
                                   marker=dict(color='rgb(0, 191, 191)', size=6)),
                      row=row, col=col)

    return fig


def _plot_params(df_features, sig, fs, times, center_e, side_e,
                 burst_params, threshold_kwargs, fig, row=1, col=1):
    """Plot bycycle burst detection parameters."""

    ylabels = [param.replace("_", " ").capitalize() for param in burst_params]

    colors = cycle(['rgba(31, 119, 180, .2)', 'rgba(214, 39, 40, .4)', 'rgba(188, 189, 34, .4)',
                    'rgba(44, 160, 44, .2)', 'rgba(148, 103, 189, .4)',
                    'rgba(255, 127, 14, .4)'])

    for idx, burst_param in enumerate(burst_params):

        # Horizontal threshold line
        thresh = threshold_kwargs[burst_param + '_threshold']

        fig.add_trace(go.Scattergl(x=[0, len(sig)/fs], y=[thresh, thresh], mode='lines',
                                    line=dict(dash="dash", color="black")),
                        row=idx+row, col=col)

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
        sstd = np.std(sig)

        sig_rects_y = np.repeat([[smax*2, smin*2, smin*2, smax*2]],
                                len(df_features), axis=0).flatten()

        fig.add_trace(go.Scattergl(x=rects_x, y=sig_rects_y, fill="toself",
                        fillcolor=fillcolor, mode="lines", line=dict(width=0)), row=1, col=col)

        # Highligh parameter subplot
        param_rects_y = np.repeat([[0, 1.1, 1.1, 0]], len(df_features), axis=0).flatten()

        fig.add_trace(go.Scattergl(x=rects_x, y=param_rects_y, fill="toself",
                      fillcolor=fillcolor, mode="lines", line=dict(width=0)), row=idx+row, col=col)

        # Burst parameter
        fig.add_trace(
            go.Scattergl(
                x=times[df_features['sample_' + center_e]], y=df_features[burst_param],
                mode='lines+markers', marker=dict(color='black')
            ),
            row=idx+row, col=col
        )

        # Axes settings
        ylabel = str(ylabels[idx] + "<br>threshold={thresh}").format(thresh=thresh)
        fig.update_yaxes(title_text=ylabel, range=[0, 1.1], dtick=.2, row=idx+row, col=col)
        fig.update_yaxes(range=[smin-2*sstd, smax+2*sstd], row=1, col=col)

        if idx == len(burst_params)-1:
            fig.update_xaxes(title_text='Time', showticklabels=True, row=idx+row, col=col)
        else:
            fig.update_xaxes(showticklabels=False, row=idx+row, col=col)

    return fig
