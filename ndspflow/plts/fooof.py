"""FOOOF plotting functions for returning ready-to-embed html."""

import re
import numpy as np
import plotly.graph_objects as go

from ndspflow.plts.utils import plot_scatter, two_column_layout


def plot_fm(fm):
    """Plot a individual FOOOF fits using plotly.

    Parameters
    ----------
    fm : fooof FOOOF
        A fooof model that has been

    Returns
    -------
    graph : str
        The fooof plot as a string containing html.
    """

    # Create figure
    fig = go.Figure()

    config = {'responsive': True}

    # Original
    fig.add_trace(go.Scatter(x=fm.freqs, y=fm.power_spectrum, mode='lines',
                                name='Original Spectrum', line=dict(color='black', width=3)))

    # Model
    fig.add_trace(go.Scatter(x=fm.freqs, y=fm.fooofed_spectrum_, mode='lines',
                                name='Full Model Fit', line=dict(color='rgba(214, 39, 40, .7)',
                                                                width=3)))

    # Aperiodic
    fig.add_trace(go.Scatter(x=fm.freqs, y=fm._ap_fit, mode='lines', name='Aperiodic Fit',
                            line=dict(color='rgba(31, 119, 180, .7)', width=3, dash='dash')))

    # Plot settings
    fig.update_layout(
        title="Spectrum Fit",
        xaxis_title="Frequency",
        yaxis_title="Power",
        font=dict(
            family="sans-serif",
            size=18,
            color='black'
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    graph = fig.to_html(full_html=False, default_height='475', default_width='700',
                        include_plotlyjs=False)

    return graph


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
    fig_pws = plot_scatter(bws, 'Peak Width', urls_peaks)

    graphs = two_column_layout(fig_bws, fig_pws, graphs)

    # AP params
    graphs.append('<br><br><center><h1>Aperiodic Parameters</h1></center>')

    fig_exp = plot_scatter(fg.get_params('aperiodic', 'exponent'), 'Exponent', urls)
    fig_off = plot_scatter(fg.get_params('aperiodic', 'offset'), 'Offset', urls, yfmt=".2f")

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
