from itertools import cycle

import numpy as np
from scipy.stats import zscore

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bycycle.utils import get_extrema_df


def plot_bm(df_features, sig, fs, threshold_kwargs, xlim=None, plot_only_result=True, use_js=True):
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

    # Create figure and subplots
    if plot_only_result and use_js:
        fig = go.Figure(make_subplots(rows=1, cols=1))
    elif plot_only_result and not use_js:
        fig = go.FigureWidget(make_subplots(rows=1, cols=1))
    elif not plot_only_result and use_js:
        fig = go.Figure(make_subplots(rows=5, cols=1, vertical_spacing=0.01))
    elif not plot_only_result and not use_js:
        fig = go.FigureWidget(make_subplots(rows=5, cols=1, vertical_spacing=0.01))

    if not plot_only_result:

        # Plot burst features
        burst_params = ['amp_fraction', 'amp_consistency', 'period_consistency', 'monotonicity',
                        'burst_fraction']
        burst_params = [param for param in burst_params if param in df_features.columns]

        ylabels = [param.replace("_", " ").capitalize() for param in burst_params]

        colors = cycle(['rgba(31, 119, 180, .2)', 'rgba(214, 39, 40, .4)', 'rgba(188, 189, 34, .4)',
                        'rgba(44, 160, 44, .2)', 'rgba(148, 103, 189, .4)',
                        'rgba(255, 127, 14, .4)'])

        for idx, burst_param in enumerate(burst_params):

            # Horizontal threshold line
            thresh = threshold_kwargs[burst_param + '_threshold']

            fig.add_trace(go.Scatter(x=[0, len(sig)/fs], y=[thresh, thresh], mode='lines',
                                     line=dict(dash="dash", color="black")),
                          row=idx+2, col=1)

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

            fig.add_trace(go.Scatter(x=rects_x, y=sig_rects_y, fill="toself",
                          fillcolor=fillcolor, mode="lines", line=dict(width=0)), row=1, col=1)

            # Highligh parameter subplot
            param_rects_y = np.repeat([[0, 1.1, 1.1, 0]], len(df_features), axis=0).flatten()

            fig.add_trace(go.Scatter(x=rects_x, y=param_rects_y, fill="toself",
                          fillcolor=fillcolor, mode="lines", line=dict(width=0)), row=idx+2, col=1)

            # Burst parameter
            fig.add_trace(
                go.Scatter(
                    x=times[df_features['sample_' + center_e]], y=df_features[burst_param],
                    mode='lines+markers', marker=dict(color='black')
                ),
                row=idx+2, col=1
            )

            # Axes settings
            ylabel = str(ylabels[idx] + "<br>threshold={thresh}").format(thresh=thresh)
            fig.update_yaxes(title_text=ylabel, range=[0, 1.1], dtick=.2, row=idx+2, col=1)
            fig.update_yaxes(range=[smin-2*sstd, smax+2*sstd], row=1, col=1)

            if idx == len(burst_params)-1:
                fig.update_xaxes(title_text='Time', showticklabels=True, row=idx+2, col=1)
            else:
                fig.update_xaxes(showticklabels=False, row=idx+2, col=1)

    # Plot signal and bursts
    for _, cyc in df_features.iterrows():

        samp_start_burst = int(cyc['sample_last_' + side_e])
        samp_end_burst = int(cyc['sample_next_' + side_e] + 1)

        times_cyc = times[samp_start_burst:samp_end_burst]
        sig_cyc = sig[samp_start_burst:samp_end_burst]

        if cyc['is_burst']:

            fig.add_trace(
                go.Scatter(x=times_cyc, y=sig_cyc, mode='lines', name="Burst",
                           line=dict(color='red', width=2)),
                row=1, col=1
            )

        else:

            fig.add_trace(
                go.Scatter(x=times_cyc, y=sig_cyc, mode='lines', name="Signal",
                           line=dict(color='black', width=2)),
                row=1, col=1
            )

    # Centers
    centers = df_features['sample_' + center_e].values
    fig.add_trace(go.Scatter(x=times[centers], y=sig[centers], mode='markers',
                             name = str(center_e.capitalize()),
                             marker=dict(color='rgb(191, 0, 191)', size=6)),
                  row=1, col=1)

    # Sides
    sides = np.append(df_features['sample_last_' + side_e].values,
                      df_features['sample_next_' + side_e].values[-1])
    fig.add_trace(go.Scatter(x=times[sides], y=sig[sides], mode='markers',
                             name = str(side_e.capitalize()),
                             marker=dict(color='rgb(0, 191, 191)', size=6)),
                  row=1, col=1)

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
        fig.update_layout(
            autosize=False,
            width=1000,
            height=1000,
            showlegend=False
        )

    else:

        fig.update_layout(
            autosize=False,
            width=1000,
            height=325,
            showlegend=False,
            yaxis_title="Voltage<br>(normalized)",
            xaxis_title="Time",
        )

    # Relabel bursts/non-bursts and convert to html
    center_times = times[df_features['sample_' + center_e].values]

    graph = relabel_bursts(fig, center_times, len(df_features),
                           len(threshold_kwargs.keys()), use_js)

    return graph


def relabel_bursts(fig, center_times, n_cycles, n_kwargs, use_js):
    """Interactively relabel bursts.

    Parameters
    ----------
    fig : plotly.graph_objects.FigureWidget
        The burst plot from :func:`~.plot_bm`.
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle.
    times : 1d array
        The time array that corresponds to the signal used to compute features.
    n_kwargs : int
        The number of threshold kwargs that were plotted.
    use_js : bool
        Uses javascript to relabel bursts if True. This is recommended when using converting the
        figure to html. Use false when plotting in a jupyter notebook.

    Returns
    -------
    fig : str or plotly.graph_objects.FigureWidget
        The fooof plot as a string containing html (when use_js=True) or a FigureWidget (when
        use_js=False).
    """

    # The number of subplots to ignore
    skip = n_kwargs * 4

    if use_js:
        # For embedding in html
        peak_trace_id = len(fig.data) - 2
        burst_traces = [idx+1 for idx in range(skip, n_cycles+skip)]

        js_callback = """
        var burstPlot = document.getElementById('{{plot_id}}')
        burstPlot.on('plotly_click', function(data){{
            curveNumber = data.points[0].curveNumber;
            burstTraces = {burst_traces};
            if (curveNumber == {trace_id}) {{
                targetTrace = burstTraces[data.points[0].pointNumber-1];
                color = burstPlot.data[targetTrace].line.color;
            }} else if (burstTraces.includes(curveNumber)) {{
                targetTrace = curveNumber;
                color = data.points[0].data.line.color;
            }} else {{
                return;
            }}
            if (color == 'black') {{
                color_inv = 'red';
            }} else {{
                color_inv = 'black';
            }}
           var update = {{'line':{{color: color_inv}}}};
           Plotly.restyle(burstPlot, update, [targetTrace]);
        }});
        """.format(trace_id=peak_trace_id, burst_traces=str(burst_traces))

        graph = fig.to_html(full_html=False, post_script=js_callback)

        return graph

    else:
        # For use in a notebook
        def _update_burst(trace, points, selector):

            if len(points.xs) == 0:
                return

            plt_idx = np.where(points.xs[0] == center_times)[0][0]

            if fig.data[skip:n_cycles+skip][plt_idx]["name"] == 'Signal':

                fig.data[skip:n_cycles+skip][plt_idx]["name"] = 'Burst'
                fig.data[skip:n_cycles+skip][plt_idx]["line"] = dict(color='red', width=2)

            elif fig.data[skip:n_cycles+skip][plt_idx]["name"] == 'Burst':

                fig.data[skip:n_cycles+skip][plt_idx]["name"] = 'Signal'
                fig.data[skip:n_cycles+skip][plt_idx]["line"] = dict(color='black', width=2)

        fig.data[-2].on_click(_update_burst)

        return fig
