from itertools import cycle

import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from fooof.plts.fm import gen_periodic

from ndspflow.core.motif import extract_motifs


def plot_motifs(fm, df_features, sig, fs, n_bursts=5, center='peak', extract_motifs_kwargs=None):
    """Plot cycle motifs using fooof fits and bycycle cycles.

    Parameters
    ----------
    fm : fooof FOOOF
        A fooof model that has been fit.
    df_features : pandas.DataFrame
        A dataframe containing bycycle features.
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    n_bursts : int, optional, default: 5
        The number of example bursts to plot per peak.
    center : {'peak', 'trough'}, optional
    extract_motifs_kwargs : dict, optional, default: None
        Keyword arguments for the :func:`~.extract_motifs` function.

    Returns
    -------
    graph : str
        The motif plot as a string containing html.
    """

    # Extract motifs
    motifs, dfs_osc = extract_motifs(fm, df_features, sig, fs, **extract_motifs_kwargs)
    motif_exists = [True if ~np.isnan(motif).all() else False for motif in motifs]

    # Initialize figure
    ncols = len(motifs)
    nrows = 2 + len(np.nonzero(motif_exists)[0])

    titles = fm.get_params('gaussian_params')[:, 0].round(1).astype(str)
    titles = [osc + ' hz Motif' for osc in titles]

    stretch_cols = [{'colspan': ncols}, *[None for _ in range(ncols-1)]]

    fig = make_subplots(
        rows=nrows, cols=ncols, row_heights=[2, 1, *[1]*(nrows-2)],
        specs=[stretch_cols, [{}] * ncols, *[stretch_cols] * int(nrows-2)],
        subplot_titles=['Spectrum & Fit', *titles, *[''] * int(nrows-2)],
        vertical_spacing=.35 / nrows
    )

    # Plot fooof
    yvals = [fm.power_spectrum, fm.fooofed_spectrum_, fm._ap_fit]
    styles = [{'color': 'black'}, {'color': '#d62728'}, {'dash': 'dash', 'color': '#1f77b4'}]
    names = ['Original', 'Full Fit', 'AP Fit']

    for yval, ls, name in zip(yvals, styles, names):

        fig.add_trace(go.Scatter(x=fm.freqs, y=yval, line=ls, name=name), row=1, col=1)

    fig.update_xaxes(title="Frequencies (hz)", row=1, col=1)
    fig.update_yaxes(title="log(Powers)", row=1, col=1)

    # Fill gaussians
    fill_colors = cycle(['rgba(44,160,44,.5)', 'rgba(255,127,14,.5)',
                         'rgba(148,103,189,.5)', 'rgba(23,190,207,.5)', 'rgba(227,119,194,.5)'])

    pe_params = fm.get_params('gaussian_params')

    colors = []
    for idx, param in enumerate(pe_params):

        fill = next(fill_colors)

        peak = fm._ap_fit + gen_periodic(fm.freqs, param)

        fig.add_trace(go.Scatter(x=np.concatenate([fm.freqs, fm.freqs[::-1]]),
                                 y=np.concatenate([peak, fm._ap_fit[::-1]]),
                                 fill='toself', fillcolor=fill, hoverinfo='none',
                                 mode='none', showlegend=False),
                      row=1, col=1)

        colors.append(fill)

    # Plot motifs and example bursting segments
    times = np.arange(0, len(sig)/fs, 1/fs)
    sig_idx = 1
    for idx, (motif, df_osc) in enumerate(zip(motifs, dfs_osc)):

        if not np.isnan(motif).any():

            # Plot motifs
            fig.add_trace(go.Scatter(x=times, y=motif, line={'color': colors[idx]},
                                     showlegend=False, hoverinfo='none'),
                          row=2, col=idx+1)

            # Plot example bursting segments
            (start, end) = _find_short_burst(df_osc, sig, n_bursts, center)

            fig.add_trace(go.Scatter(x=times[start:end], y=sig[start:end],
                                     line={'color': colors[idx]}, showlegend=False),
                          row=2+sig_idx, col=1, )

            fig.update_xaxes(title_text='Time (s)', row=2+sig_idx, col=1)
            fig.update_yaxes(title_text='Normalized Voltage', row=2+sig_idx, col=1)
            sig_idx += 1

        else:

            # Plot text for no detected oscillations
            fig.add_trace(
                go.Scatter(x=[0], y=[0],
                    mode="text",
                    text=["No<br>Oscillation<br>Found"],
                    textposition="middle center",
                    textfont=dict(
                        size=24,
                        color=colors[idx].replace('.5', '.75')
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ),
                row=2, col=idx+1
            )

        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=idx+1)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=idx+1)

    # Set layout
    fig.update_layout(
        autosize=True,
        width=900,
        height=250 * nrows,
        showlegend=True
    )

    graph = fig.to_html(full_html=False, include_plotlyjs=False)

    return graph


def _find_short_burst(df_features, sig, n_bursts=5, center='peak'):
    """Find n consectutive bursting sample locations.

    Parameters
    ----------
    df_features : pandas.DataFrame
        A dataframe containing bycycle features.
    sig : 1d array
        Time series.
    n_bursts : int
        The length, in samples, of the representative burst.
    center : {'peak', 'trough'}, optional
        The center definition of cycles.

    Returns
    -------
    locs : tuple of (int, int)
        The sample location (indices) of the first n consectuive bursts.
    """

    # Get indices of non-consecutive samples
    indices = df_features.index.values
    diffs = np.diff(indices) != 1
    diffs = np.nonzero(diffs)[0] + 1

    # Split samples into bursting segments
    bursts = np.split(indices, diffs)

    # Get the longest burst and liit to n_bursts
    burst = bursts[np.argmax(np.array([len(arr) for arr in bursts]))]
    burst = burst[:n_bursts]

    side = 'trough' if center == 'peak' else 'peak'
    locs = (df_features['sample_last_' + side][burst[0]],
            df_features['sample_next_' + side][burst[-1]])

    return locs
