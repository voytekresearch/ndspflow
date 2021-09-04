
"""Motif plotting functions."""

import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from neurodsp.utils.norm import normalize_sig

from ndspflow.plts.fooof import plot_fm
from ndspflow.motif import extract


def plot_motifs(motif, n_bursts=5, center='peak', normalize=True,
                plot_fm_kwargs=None, plot_sig=True):
    """Plot cycle motifs using fooof fits and bycycle cycles.

    Parameters
    ----------
    motif : ndspflow.motif.Motif
        Motif object that has been fit.
    n_bursts : int, optional, default: 5
        Max number of example bursts to plot per peak.
    center : {'peak', 'trough'}, optional
        Defines centers of bycycle cycles.
    normalize : book, optiona, default: True
        Signal is mean centered with variance of one if True.
    plot_fm_kwargs : dict, optional, default: None
        Keyword arguments for the :func:`~.plot_fm` function.
    plot_sig : bool, optional, default: True
        Whether to plot example segments from the signal.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A plotly figure of the spectrum, motif(s), and signal.
    """

    # Extract required attributes
    fm = motif.fm
    sig = motif.sig
    sig = normalize_sig(sig, mean=0, variance=1) if normalize else sig
    fs = motif.fs
    dfs_features = [result.df_features for result in motif]
    results = [result.motif for result in motif]
    labels = [result.labels for result in motif]

    # Initialize figure and settings
    ncols = len(results)

    nrows = 2

    if plot_sig:
        # Add extra rows for example signals
        for result in results:
            if isinstance(result, np.ndarray):
                nrows += 1
            elif isinstance(result, list):
                for sub_result in result:
                    if isinstance(sub_result, np.ndarray):
                        nrows+=1

    specs = [
        [{'colspan': ncols, 'b': .4/nrows}, *[None] * (ncols-1)],
        [{'b': .1/nrows} for _ in range(ncols)],
        *[[{'colspan': ncols, 'b': .1/nrows}, *[None] * (ncols-1)]] * (nrows-2)
    ]

    cfs = fm.get_params('peak', 'CF')
    cfs = [round(cfs, 1)] if not isinstance(cfs, np.ndarray) else cfs.round(1)
    titles = [str(cf) + ' hz Motif' for cf in cfs]

    row_heights = [2, 1, *[1] * (nrows-2)]

    fig = make_subplots(
        rows=nrows, cols=ncols, row_heights=row_heights, specs=specs,
        subplot_titles=['Spectrum Fit', *titles, *[''] * int(nrows-2)],
        vertical_spacing=.03 / nrows
    )

    # Plot fooof
    plot_fm_kwargs = {} if plot_fm_kwargs is None else plot_fm_kwargs

    default_fills = ['rgba(44,160,44,.5)', 'rgba(255,127,14,.5)',
                     'rgba(148,103,189,.5)', 'rgba(23,190,207,.5)', 'rgba(227,119,194,.5)']

    fill_gaussians = plot_fm_kwargs.pop('fill_gaussians', default_fills)
    log_freqs = plot_fm_kwargs.pop('log_freqs', False)

    fooof_fig = plot_fm(fm, fill_gaussians=fill_gaussians, log_freqs=log_freqs, **plot_fm_kwargs)
    for trace in fooof_fig.select_traces():
        fig.add_trace(trace, row=1, col=1)

    # Label axes
    xaxis_title = 'log(Frequencies)' if log_freqs else 'Frequencies'
    xaxis_title = plot_fm_kwargs.pop('xaxis_title', xaxis_title)
    yaxis_title = plot_fm_kwargs.pop('yaxis_title', 'log(Power)')

    fig.update_xaxes(title_text=xaxis_title, row=1, col=1)
    fig.update_yaxes(title_text=yaxis_title, row=1, col=1)

    # Vertically stack
    if sig.ndim == 1:

        sig = sig.reshape(1, len(sig))

        sig = np.repeat(sig, len(results), axis=0)

    # Plot motifs and example bursting segments
    times = np.arange(0, len(sig[0])/fs, 1/fs)

    # Iterate over each center freq
    row_idx = 1
    for result_idx, (result, df_osc) in enumerate(zip(results, dfs_features)):

        color = default_fills[result_idx % len(default_fills)]
        color = color.replace('.5', '1')

        if isinstance(result, float):
            _plot_motif(fig, color, result_idx+1, null=True)
            continue

        plt_none = 0
        for motif_idx, motif in enumerate(result):

            if isinstance(motif, float):
                plt_none += 1
                continue

            # Plot motif waveforms
            fig.add_trace(go.Scatter(x=times, y=motif, line={'color': color},
                                        mode='lines', showlegend=False, hoverinfo='none'),
                        row=2, col=result_idx+1)

            _plot_motif(fig, color, result_idx+1, False, x=times, y=motif)

            # Split dataframe by cluster
            if not plot_sig:
                continue

            if not isinstance(labels[result_idx], float):
                # Multiple clusters
                sub_labels = np.where(labels[result_idx] == motif_idx)
                df = df_osc.iloc[sub_labels] if len(sub_labels) != 0 else df_osc
            else:
                # Single cluster
                df = df_osc

            (start, end) = _find_short_burst(df, n_bursts, center)

            fig.add_trace(go.Scatter(x=times[start:end], y=sig[result_idx][start:end],
                                     line={'color': color}, showlegend=False),
                          row=2+row_idx, col=1)

            row_idx += 1

            fig.update_yaxes(title_text='Normalized Voltage', row=1+row_idx, col=1)

        if plt_none == len(result) and row_idx > 1:
            _plot_motif(fig, color, result_idx+1, null=True)

    if plot_sig:
        fig.update_xaxes(title_text='Time (s)', row=1+row_idx, col=1)

    # Set layout
    fig.update_layout(
        autosize=True,
        width=900,
        height=250 * nrows,
        showlegend=True
    )

    return fig


def _find_short_burst(df_features, n_bursts=5, center='peak'):
    """Find n consectutive bursting sample locations.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Dataframe containing bycycle features.
    n_bursts : int
        Maximum number of consecutive cycles to plot.
    center : {'peak', 'trough'}, optional
        Center definition of cycles.

    Returns
    -------
    locs : tuple of (int, int)
        Sample location (indices) of the first n consectuive bursts.
    """

    # Get indices of non-consecutive samples
    indices = df_features.index.values
    diffs = np.diff(indices) != 1
    diffs = np.nonzero(diffs)[0] + 1

    # Split samples into bursting segments
    bursts = np.split(indices, diffs)

    # Get the longest burst and limit to n_bursts
    burst = bursts[np.argmax(np.array([len(arr) for arr in bursts]))]
    burst = burst[:n_bursts]

    side = 'trough' if center == 'peak' else 'peak'
    locs = (df_features['sample_last_' + side][burst[0]],
            df_features['sample_next_' + side][burst[-1]])

    return locs


def _plot_motif(fig, color, col, null=True, x=None, y=None):
    """Plot a noramlized motif."""

    # Plot text for peaks with no detected oscillations
    if null:
        fig.add_trace(
            go.Scatter(x=[0], y=[0],
                mode="text",
                text=["No<br>Oscillation<br>Found"],
                textposition="middle center",
                textfont=dict(
                    size=18,
                    color=color.replace('.5', '.75')
                ),
                showlegend=False,
                hoverinfo='none'
            ),
            row=2, col=col
        )
    else:
        fig.add_trace(go.Scatter(x=x, y=y, line={'color': color},
                                 mode='lines', showlegend=False, hoverinfo='none'),
                        row=2, col=col)

    fig.update_xaxes(showticklabels=False, showgrid=False,
                     zeroline=False, row=2, col=col)

    fig.update_yaxes(showticklabels=False, showgrid=False,
                     zeroline=False, row=2, col=col)
