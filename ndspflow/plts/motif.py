
"""Motif plotting functions."""

import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from neurodsp.utils.norm import normalize_sig

from ndspflow.plts.fooof import plot_fm
from ndspflow.motif import extract


def plot_motifs(motif, n_bursts=5, center='peak', normalize=True, plot_fm_kwargs=None):
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
    dfs_features = [result.df_features for result in motif.results]
    results = [result.motif for result in motif.results]

    # Get indices where motifs are found with greater than 1 cycle
    motif_exists = ~np.array([isinstance(result, float) for result in results])

    drop = [idx for idx in np.where(motif_exists)[0] if len(dfs_features[idx]) <= 1]
    motif_exists[drop] = False

    # Initialize figure
    ncols = len(results)

    nrows = 2
    for idx, result in enumerate(results):
        if motif_exists[idx]:
            nrows += len(result)

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

    # Plot motifs and example bursting segments
    times = np.arange(0, len(sig)/fs, 1/fs)

    motif_idxs = np.where(motif_exists)[0]
    last_motif_idx = motif_idxs[-1] if len(motif_idxs) > 0 else None

    # Iterate over each center freq
    row_idx = 1
    for idx, (result, df_osc) in enumerate(zip(results, dfs_features)):

        color = default_fills[idx % len(default_fills)]
        color = color.replace('.5', '1')

        if idx in motif_idxs:

            # Plot mean waveforms
            for sub_motif in result:

                # Plot motifs
                fig.add_trace(go.Scatter(x=times, y=sub_motif, line={'color': color}, mode='lines',
                                        showlegend=False, hoverinfo='none'),
                              row=2, col=idx+1)

            # Plot example bursting segments
            for midx in range(len(motif[idx].motif)):

                if not isinstance(motif[idx].labels, float):
                    # Multi cluster
                    labels = np.where(motif[idx].labels == midx)
                    df = df_osc.iloc[labels] if len(labels) != 0 else df_osc
                else:
                    # Single cluster
                    df = df_osc

                (start, end) = _find_short_burst(df, n_bursts, center)

                fig.add_trace(go.Scatter(x=times[start:end], y=sig[start:end],
                                        line={'color': color}, showlegend=False),
                              row=2+row_idx, col=1)

                row_idx += 1

            # Label axes
            if idx == last_motif_idx:

                fig.update_xaxes(title_text='Time (s)', row=1+row_idx, col=1)

                fig.update_yaxes(title_text='Normalized Voltage', row=1+row_idx, col=1)

        else:

            # Plot text for peaks with no detected oscillations
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
