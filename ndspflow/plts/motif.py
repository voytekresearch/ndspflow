
"""Motif plotting functions."""

import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from neurodsp.utils.norm import normalize_sig

from ndspflow.plts.fooof import plot_fm
from ndspflow.core.motif import extract_motifs


def plot_motifs(fm, df_features, sig, fs, n_bursts=5, center='peak', normalize=True,
                motifs=None, cycles=None, extract_motifs_kwargs=None, plot_fm_kwargs=None):
    """Plot cycle motifs using fooof fits and bycycle cycles.

    Parameters
    ----------
    fm : fooof FOOOF or tuple
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
        Defines centers of bycycle cycles.
    normalize : book, optiona, default: True
        Signal is mean centered with variance of one if True.
    extract_motifs_kwargs : dict, optional, default: None
        Keyword arguments for the :func:`~.extract_motifs` function.
    plot_fm_kwargs : dict, optional, default: None
        Keyword arguments for the :func:`~.plot_fm` function.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A plotly figure of the spectrum, motif(s), and signal.
    """

    # Extract motifs
    sig = normalize_sig(sig, mean=0, variance=1) if normalize else sig
    extract_motifs_kwargs = {} if extract_motifs_kwargs is None else extract_motifs_kwargs

    if motifs is None or cycles is None:

        motifs, cycles = extract_motifs(fm, df_features, sig, fs, return_cycles=True,
                                        **extract_motifs_kwargs)

    # Get indices where motifs are found with greater than 1 cycle
    dfs_osc = cycles['dfs_osc']
    motif_exists = ~np.array([isinstance(motif, float) for motif in motifs])
    drop = [idx for idx in np.where(motif_exists)[0] if len(dfs_osc[idx]) <= 1]
    motif_exists[drop] = False

    # Initialize figure
    ncols = len(motifs)
    nrows = len(np.nonzero(motif_exists)[0]) + 2

    specs = [
        [{'colspan': ncols, 'b': .4/nrows}, *[None] * (ncols-1)],
        [{'b': .1/nrows} for _ in range(ncols)],
        *[[{'colspan': ncols, 'b': .1/nrows}, *[None] * (ncols-1)]] * (nrows-2)
    ]

    titles = [str(round(fs/len(motif[0]))) for motif in motifs if not isinstance(motif, float)]
    titles = [osc + ' hz Motif' for osc in titles]

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
    xaxis_title = plot_fm_kwargs.pop('xaxis_title', 'Frequencies')
    yaxis_title = plot_fm_kwargs.pop('yaxis_title', 'log(Power)')

    fig.update_xaxes(title_text=xaxis_title, row=1, col=1)
    fig.update_yaxes(title_text=yaxis_title, row=1, col=1)

    # Plot motifs and example bursting segments
    times = np.arange(0, len(sig)/fs, 1/fs)

    motif_idxs = np.where(motif_exists)[0]
    last_motif_idx = motif_idxs[-1] if len(motif_idxs) > 0 else None

    # Iterate over each center freq
    row_idx = 1
    for idx, (motif, df_osc) in enumerate(zip(motifs, dfs_osc)):

        color = default_fills[idx % len(default_fills)]

        if idx in motif_idxs:

            # Iterate over motif(s) at each center freq
            for sub_motif in motif:

                # Plot motifs
                fig.add_trace(go.Scatter(x=times, y=sub_motif, line={'color': color}, mode='lines',
                                        showlegend=False, hoverinfo='none'),
                            row=2, col=idx+1)

                # Plot example bursting segments
                (start, end) = _find_short_burst(df_osc, n_bursts, center)

                fig.add_trace(go.Scatter(x=times[start:end], y=sig[start:end],
                                        line={'color': color}, showlegend=False),
                            row=2+row_idx, col=1)

                if idx == last_motif_idx:
                    fig.update_xaxes(title_text='Time (s)', row=2+row_idx, col=1)

                fig.update_yaxes(title_text='Normalized Voltage', row=2+row_idx, col=1)

            row_idx += 1

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
        A dataframe containing bycycle features.
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

    # Get the longest burst and limit to n_bursts
    burst = bursts[np.argmax(np.array([len(arr) for arr in bursts]))]
    burst = burst[:n_bursts]

    side = 'trough' if center == 'peak' else 'peak'
    locs = (df_features['sample_last_' + side][burst[0]],
            df_features['sample_next_' + side][burst[-1]])

    return locs
