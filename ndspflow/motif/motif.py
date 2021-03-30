"""Extract motifs from frequency ranges."""

import warnings
import numpy as np
import pandas as pd

from ndspflow.core.utils import limit_df
from ndspflow.core.fit import fit_bycycle

from ndspflow.motif.cluster import cluster_cycles
from ndspflow.motif.burst import motif_burst_detection
from ndspflow.motif.utils import split_signal


def robust_extract(fm, sig, fs, clust_score=0.5, corr_thresh=5., center='peak'):
    """Extract bursting using motifs after motif burst detection.

    Parameters
    ----------
    fm : fooof.FOOOF or list of tuple
        A fooof model that has been fit, or a list of (center_freq, bandwidth).
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    clust_score : float, optional, default: 0.5
        The silhouette score to accept k clusters.
    corr_thresh : float, optional, default: 5
        Cross-correlation coefficient threshold.
    center : {'peak', 'trough'}, optional
        The center definition of cycles.

    Returns
    -------
    motifs : list of list of 1d arrays
        Motifs for each center frequency in ascending order. Inner list contains multiple arrays
        if multiple motifs are found at one frequency.
    cycles : dict
        The timeseries, dataframes, frequency ranges, and predicted labels for each cycle.
        Valid keys include: 'sigs', 'dfs_osc', 'labels', 'f_ranges'.
        Only returned when ``return_cycles`` is True.
    """

    # First pass motif extraction
    motifs_, cycles_ = extract(fm, sig, fs, clust_score=clust_score, center=center,
                               only_bursts=False, max_clusters=10)

    motifs = []
    cycles = {'sigs': [], 'dfs_osc': [], 'labels': [], 'f_ranges': []}

    for idx, (motif, f_range) in enumerate(zip(motifs_, cycles_['f_ranges'])):

        # Skip null motifs (np.nan)
        if isinstance(f_range, float):
            motifs.append(np.nan)
            for key in cycles:
                cycles[key].append(np.nan)
            continue

        # Motif correlation burst detection
        bm = fit_bycycle(sig, fs, f_range)

        is_burst = motif_burst_detection(motif, bm, sig, corr_thresh=corr_thresh)

        bm['is_burst'] = is_burst

        # Re-extract motifs from bursts
        motifs_burst, cycles_burst = extract(fm, sig, fs, df_features=bm, clust_score=clust_score,
                                             center=center, only_bursts=True)

        if isinstance(motifs_burst[0], float):
            motifs.append(np.nan)
            for key in cycles:
                cycles[key].append(np.nan)
            continue

        motifs.append(motifs_burst[0])

        # Collect cycles

        cycles['sigs'].append(cycles_burst['sigs'][idx])
        cycles['dfs_osc'].append(cycles_burst['dfs_osc'][idx])
        cycles['labels'].append(cycles_burst['labels'][idx])
        cycles['f_ranges'].append(cycles_burst['f_ranges'][idx])

    return motifs, cycles


def extract(fm, sig, fs, df_features=None, scaling=1, only_bursts=True,
            center='peak', clust_score=1, max_clusters=10, min_n_cycles=10):
    """Get the average cycle from a bycycle dataframe for all fooof peaks.

    Parameters
    ----------
    fm : fooof.FOOOF or list of tuple
        A fooof model that has been fit, or a list of (center_freq, bandwidth).
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    df_features : pandas.DataFrame, optional, default: None
        A dataframe containing bycycle features.
    scaling : float, optional, default: 1
        The scaling of the bandwidth from the center frequencies to limit cycles to.
    only_burst : bool, optional, default: True
        Limits the dataframe to bursting cycles when True.
    center : {'peak', 'trough'}, optional
        The center definition of cycles.
    clust_score : float, optional, default: 1
        The silhouette score to accept k clusters..
    max_clusters : int, optional, default: 10
        The maximum number of clusters to evaluate.
    min_n_cycles : int, optional, default: 10
        The minimum number of cycles required to be considered at motif.

    Returns
    -------
    motifs : list of list of 1d arrays
        Motifs for each center frequency in ascending order. Inner list contains multiple arrays
        if multiple motifs are found at one frequency.
    cycles : dict
        The timeseries, dataframes, frequency ranges, and predicted labels for each cycle.
        Valid keys include: 'sigs', 'dfs_osc', 'labels', 'f_ranges'.
        Only returned when ``return_cycles`` is True.
    """

    # Extract center freqs and bandwidths from fooof fit
    if not isinstance(fm, list):

        cfs = fm.get_params('peak_params', 'CF')
        bws = fm.get_params('peak_params', 'BW')
        cfs = cfs if isinstance(cfs, (list, np.ndarray)) else [cfs]
        bws = bws if isinstance(bws, (list, np.ndarray)) else [bws]

    else:

        cfs = np.array(fm)[:, 0]
        bws = np.array(fm)[:, 1]

    f_ranges = [(cf-(scaling * bws[idx]), cf+(scaling * bws[idx])) for idx, cf in enumerate(cfs)]

    # Get cycles within freq ranges
    motifs = []
    cycles = {'sigs': [], 'dfs_osc': [], 'labels': [], 'f_ranges': []}

    for f_range in f_ranges:

        if df_features is None:
            df_features = fit_bycycle(sig, fs, f_range, center)

        # Restrict dataframe to frequency range
        df_osc = limit_df(df_features, fs, f_range, only_bursts=only_bursts)

        # No cycles found in frequency range
        if not isinstance(df_osc, pd.DataFrame):
            motifs.append(np.nan)
            for key in cycles:
                cycles[key].append(np.nan)
            continue

        # Split signal into 2d array of cycles
        sig_cyc = split_signal(df_osc, sig, True, center)

        # Cluster cycles
        labels = cluster_cycles(sig_cyc, clust_score=clust_score, max_clusters=max_clusters)

        if len(sig_cyc) <  min_n_cycles:
            continue

        if not isinstance(labels, np.ndarray):
            # No clusters found
            motifs.append([np.mean(sig_cyc, axis=0)])
        else:
            # Multiple motifs found at the current frequency range
            multi_motifs = []
            for idx in range(max(labels)+1):
                multi_motifs.append(np.mean(sig_cyc[np.where(labels == idx)[0]], axis=0))
            motifs.append(multi_motifs)

            # Recompute labels using cross-correlation coefficient
            labels = np.zeros_like(labels)

            for idx, cyc in enumerate(sig_cyc):

                corrs = []
                for motif in multi_motifs:
                    corrs.append(np.correlate(motif, cyc, mode='valid')[0])

                labels[idx] = np.argmax(corrs)

        # Collect cycles
        cycles['sigs'].append(sig_cyc)
        cycles['dfs_osc'].append(df_osc)
        cycles['labels'].append(labels)
        cycles['f_ranges'].append(f_range)

    return motifs, cycles
