"""FOOOF plotting functions."""

from itertools import cycle
import numpy as np
import plotly.graph_objects as go

from fooof.plts.fm import gen_periodic

from ndspflow.plts.utils import plot_scatter, two_column_layout


def plot_fm(fm, log_freqs=False, fill_gaussians=False, showlegend=True, **kwargs):
    """Plot a single FOOOF fits using plotly.

    Parameters
    ----------
    fm : fooof FOOOF
        A fooof model that has been fit.
    log_freqs : bool, optional, default: False
        Logs frequencies when True.
    fill_gaussians : bool or list, optional, default: False
        Shades gaussians when True. A list of plotly colors, either hex or rgba, may passed to
        control the color of each shade. The default color is green.
    showlegend : bool, optional, default: True
        Show the plot legend when True.
    **kwargs
        Additional keyword arguments to pass to the ``update_layout`` method of a plotly figure.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A plotly figure of the spectrum and fit.
    """

    # Create figure
    fig = go.Figure()

    # Plot traces
    freqs = np.log10(fm.freqs) if log_freqs else fm.freqs

    y_traces = [fm.power_spectrum, fm.fooofed_spectrum_, fm._ap_fit]

    styles = [{'color': 'black'}, {'color': '#d62728'}, {'dash': 'dash', 'color': '#1f77b4'}]
    names = ['Original', 'Full Fit', 'Aperiodic Fit']

    for y_trace, ls, name in zip(y_traces, styles, names):

        fig.add_trace(go.Scatter(x=freqs, y=y_trace, line=ls, name=name, showlegend=showlegend))

    # Fill gaussians
    if fill_gaussians is not False:

        if isinstance(fill_gaussians, list):

            # Custom gaussian colors
            fill_colors = cycle(fill_gaussians)

        else:

            # Default gaussian colors
            fill_colors = cycle(['rgba(44,160,44,.5)'])

        pe_params = fm.get_params('gaussian_params')

        for param in pe_params:

            fill = next(fill_colors)

            peak = fm._ap_fit + gen_periodic(fm.freqs, param)

            fig.add_trace(go.Scatter(x=np.concatenate([freqs, freqs[::-1]]),
                                     y=np.concatenate([peak, fm._ap_fit[::-1]]),
                                     fill='toself', fillcolor=fill, hoverinfo='none',
                                     mode='none', showlegend=False))

    # Update kwargs
    fig.update_xaxes(title="log(Frequencies)" if log_freqs else "Frequencies")
    fig.update_yaxes(title="log(Power)")

    title = kwargs.pop('title', "Spectrum Fit")

    fig.update_layout(title=title, **kwargs)

    return fig


def plot_fg(fg, urls):
    """Plot FOOOFGroup parameters distributions using plotly.

    Parameters
    ----------
    fg : fooof FOOOFGroup
        FOOOFGroup object that have been fit using :func:`ndspflow.core.fit.fit_fooof`.
    urls : list of str
        Local html paths to link points to their individual reports.

    Returns
    -------
    graphs : str
        Multiple FOOOFGroup plots as a string containing html.
    """

    graphs = []

    # Fit
    graphs.append('<br><br><center><h1>Goodness of Fit</h1></center>')

    fig_error = plot_scatter(fg.get_params('error'), 'Error', urls)
    fig_rsq = plot_scatter(fg.get_params('r_squared'), 'R-Squared', urls)

    graphs = two_column_layout(fig_error, fig_rsq, graphs)

    # PE params
    peak_idx = fg.get_params('peak_params', 'CF')[:, 1].astype(int)

    urls_peaks = [urls[idx] for idx in peak_idx]

    graphs.append('<br><br><center><h1>Peak Parameters</h1></center>')

    _, n_peaks = np.unique(peak_idx, return_counts=True)
    cfs = fg.get_params('peak_params', 'CF')[:, 0]

    fig_npeaks = plot_scatter(n_peaks, 'Number of Peaks', urls)
    fig_cf = plot_scatter(cfs, 'Center Frequency', urls_peaks)

    graphs = two_column_layout(fig_npeaks, fig_cf, graphs)

    bws = fg.get_params('peak_params', 'BW')[:, 0]
    pws = fg.get_params('peak_params', 'PW')[:, 0]

    fig_bws = plot_scatter(bws, 'Band Width', urls_peaks)
    fig_pws = plot_scatter(pws, 'Peak Width', urls_peaks)

    graphs = two_column_layout(fig_bws, fig_pws, graphs)

    # AP params
    graphs.append('<br><br><center><h1>Aperiodic Parameters</h1></center>')

    fig_exp = plot_scatter(fg.get_params('aperiodic', 'exponent'), 'Exponent', urls)
    fig_off = plot_scatter(fg.get_params('aperiodic', 'offset'), 'Offset', urls)

    graphs = two_column_layout(fig_exp, fig_off, graphs)

    # Place each item in the list on a newline
    graphs = "\n".join(graphs)

    return graphs


def plot_fgs(fgs, urls):
    """Plot a list of FOOOFGroup parameters distributions using plotly.

    Parameters
    ----------
    fgs : list of fooof FOOOFGroup
        FOOOFGroup objects that have been fit using :func:`ndspflow.core.fit.fit_fooof`.
    urls : list of str
        Local html paths to link points to their individual reports.

    Returns
    -------
    graphs : list of str
        Multiple FOOOFGroup plots as a string containing html.
    """

    graphs = []

    urls_reshape = np.reshape(urls, (len(fgs), int(len(urls)/len(fgs))))

    for idx, (fg, url) in enumerate(zip(fgs, urls_reshape)):

        graphs.append(
            "<br><br><button type=\"button\" class=\"collapsible\">"\
            "<h1 style=\"text-align:left\">▼ Group Index: {idx}</h1></button>"\
            "<div class=\"content\">"\
            .format(idx=str(idx).zfill(4))
        )

        graphs.append(plot_fg(fg, url) + "</div>")

    return graphs
