"""Bycycle plotting functions for returning ready-to-embed html."""

from itertools import cycle
import re

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

    fig.update_layout(autosize=False, showlegend=False, title_text="Burst Detection Plots")

    graph = fig.to_html(include_plotlyjs=False)

    # Get data as a list to pass into js array
    columns = df_features.columns.values.tolist()
    plot_data = df_features.values.astype('str').tolist()
    plot_data.insert(0, columns)

    # Get burst traces
    trace_id = len(fig.data) - 2
    burst_traces = [idx for idx in range(0, len(df_features))]

    # Get the div id containg the plot
    div_id = re.search("<div id=.+?\"", graph)[0]
    div_id = div_id[9:-1]

    # Create js callback
    js_callback = ["<script type=\"text/javascript\">\n"]
    js_callback.append(_relabel_bursts())
    js_callback.append("""
    var burstPlot = document.getElementById('{plot_id}');
    var burstTraces = {burst_traces};
    var traceId = {trace_id};
    var plotData = {plot_data};
    burstPlot.on('plotly_click', function(data){{
        relabelBursts(data, burstPlot, plotData, burstTraces, traceId)
    }});
    """.format(trace_id=trace_id, burst_traces=str(burst_traces),
                plot_data=str(plot_data), plot_id=div_id)
    )
    js_callback.append("</script>\n")
    js_callback = "".join(js_callback)

    # Flatten lists into single string
    graph = re.sub("</body>\n</html>", "\n", graph)

    # Add recompute burst btn below the plots
    btn = "\n\t\t<p><center><button onclick=\"saveCsv(plotData)\" class=\"btn\" "
    btn = btn + "title=\"update is_burst column\">Update Bursts</button></center></p>"

    graph = graph + js_callback + btn + "\n</body>\n</html>"

    return graph


def plot_bg(dfs_features, sigs, fs, xlim=None):
    """Plot 2D bycycle fits.

    Parameters
    ----------
    dfs_features : list of pandas.DataFrame
        Dataframe scontaining shape and burst features for each cycle.
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

    # Initialize figures in groups of 10
    #   Plotly doesn't render single figures well with 100+ plots
    n_figs = int(np.ceil(len(sigs) / 10))
    figs = np.zeros(n_figs).tolist()

    titles = []
    for idx in range(n_figs):

        # Subplot titles
        start = idx * 10
        end = start + len(sigs[start:start+10])
        titles.append("Indices: {start}-{end}".format(start=start, end=end))

        # Create subplots
        fig = make_subplots(rows=10, cols=1, vertical_spacing=0, shared_xaxes=True)
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
        fig_idx = int(np.ceil((idx+1)/10)) - 1
        row_idx = int(idx - (10*fig_idx))

        _plot_bursts(df_features, sig, times, center_e, side_e,
                    figs[fig_idx], row=row_idx+1, col=1)

    graphs = []
    for idx, fig in enumerate(figs):

        # Update the figures
        figs[idx].update_layout(
            autosize=False,
            width=1000,
            height=1000,
            showlegend=False,
            margin_autoexpand=False,
            title_text=titles[idx]
        )

        # Convert to html
        if idx == 0:

            graphs.append(fig.to_html(include_plotlyjs=True))

        else:

            div = re.search("<div>.*</div>", fig.to_html(include_plotlyjs=False))[0]
            graphs.append(div + "\n")

    # Custom js callback
    js_callback = ["<script type=\"text/javascript\">\n"]
    js_callback.append(_relabel_bursts())

    peaks = np.zeros((n_figs, 10)).tolist()
    start_traces = np.zeros((n_figs, 10)).tolist()
    for idx, fig in enumerate(figs):

        peaks[idx] = [tidx for tidx, trace in enumerate(fig.data)
                      if trace.name == 'Peak']

        start_traces[idx] = np.insert(np.array(peaks[idx])[:-1] + 2, 0, 0)

    for idx, df_feature in enumerate(dfs_features):

        # Get the div id containg the plot
        graph_idx = int(np.ceil((idx+1)/10)) - 1
        graph = graphs[graph_idx]
        div_id = re.search("<div id=.+?\"", graph)[0]
        div_id = div_id[9:-1]

        # Get the peak trace index
        trace_idx = int(idx - 10 * (np.ceil((idx+1) / 10) - 1))
        trace_id = peaks[graph_idx][trace_idx]

        # Get signal/burst trace indices
        burst_traces = np.array([idx for idx in range(0, len(df_features))])
        burst_traces = burst_traces + start_traces[graph_idx][trace_idx]
        burst_traces = burst_traces.tolist()

        # Get data as a list to pass into js array
        columns = df_features.columns.values.tolist()
        plot_data = df_features.values.astype('str').tolist()
        plot_data.insert(0, columns)

        # Callback for every plot
        js_callback.append("""
        var burstPlot{idx} = document.getElementById('{plot_id}');
        var burstTraces{idx} = {burst_traces};
        var traceId{idx} = {trace_id};
        var plotData{idx} = {plot_data};
        burstPlot{idx}.on('plotly_click', function(data){{
            relabelBursts(data, burstPlot{idx}, plotData{idx}, burstTraces{idx}, traceId{idx})
        }});
        try{{
            allData.push(plotData{idx})
        }} catch(e){{
            allData = plotData{idx}
        }}
        """.format(trace_id=trace_id, burst_traces=str(burst_traces),
                   plot_data=str(plot_data), plot_id=div_id, idx=str(idx))
        )

    js_callback.append("</script>\n")

    # Flatten lists into single string
    graphs.append("".join(js_callback))

    btn = "\n\t\t<p><center><button onclick=\"saveCsv(allData)\" class=\"btn\" "
    btn = btn + "title=\"update is_burst column\">Update Bursts</button></center></p>"
    graphs.append(btn)

    graphs[0] = re.sub("</body>\n</html>", "", graphs[0])
    graphs.append("\n</body>\n</html>")
    graphs = "".join(graphs)

    return graphs


def _relabel_bursts():
    """Generate javascript that is used to relabel bursts."""

    js_callback = """
    function relabelBursts(data, burstPlot, plotData, burstTraces, traceId) {
        var curveNumber = data.points[0].curveNumber;
        if (curveNumber == traceId) {
            var targetTrace = burstTraces[data.points[0].pointNumber];
            var color = burstPlot.data[targetTrace].line.color;
        } else if (burstTraces.includes(curveNumber)) {
            var targetTrace = curveNumber;
            var color = data.points[0].data.line.color;
        } else {
            return;
        }
        if (color == 'black') {
            var color_inv = 'red';
            var isBurst = 'True';
        } else {
            var color_inv = 'black';
            var isBurst = 'False';
        }
        var update = {'line':{color: color_inv}};
        Plotly.restyle(burstPlot, update, [targetTrace]);
        cyc = targetTrace-burstTraces[0];
        if ( ! plotData[0].includes('is_burst_new')){
            plotData[0][plotData[0].length-1] = 'is_burst_orig';
            plotData[0].push('is_burst_new');
            for (i=1; i<plotData.length; i++){
                plotData[i][plotData[0].length-1] = plotData[i][plotData[0].length-2]
            }
        }
        plotData[cyc+1][plotData[cyc].length-1] = isBurst;
    }
    """

    return js_callback

def _plot_bursts(df_features, sig, times, center_e, side_e, fig, row=1, col=1):
    """Plot where a signal is bursting"""

    # Plot signal and bursts
    for _, cyc in df_features.iterrows():

        samp_start_burst = int(cyc['sample_last_' + side_e])
        samp_end_burst = int(cyc['sample_next_' + side_e] + 1)

        times_cyc = times[samp_start_burst:samp_end_burst]
        sig_cyc = sig[samp_start_burst:samp_end_burst]

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
