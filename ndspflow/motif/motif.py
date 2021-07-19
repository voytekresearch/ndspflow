"""Extract motifs from frequency ranges."""

import numpy as np
import pandas as pd

from ndspflow.core.utils import limit_df
from ndspflow.core.fit import fit_bycycle

from ndspflow.motif.cluster import cluster_cycles

from ndspflow.motif.utils import split_signal



def extract(fm, sig, fs, df_features=None, scaling=1, use_thresh=True, center='peak',
            min_clust_score=1, var_thresh=0.05, min_clusters=2, max_clusters=10, min_n_cycles=10,
            random_state=None):
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
    use_thresh : bool, optional, default: True
        Limits the dataframe to super variance and correlation thresholds.
    center : {'peak', 'trough'}, optional
        The center definition of cycles.
    min_clust_score : float, optional, default: 1
        The minimum silhouette score to accept k clusters. The default skips clustering.
    var_thresh : float, optional, default: 0.05
        Height threshold in variance.
    min_clusters : int, optional, default: 2
        The minimum number of clusters to evaluate.
    max_clusters : int, optional, default: 10
        The maximum number of clusters to evaluate.
    min_n_cycles : int, optional, default: 10
        The minimum number of cycles required to be considered at motif.
    random_state : int, optional, default: None
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic for reproducible results.

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
    cycles = {'sigs': [], 'dfs_features': [], 'labels': [], 'f_ranges': []}

    for f_range in f_ranges:

        if df_features is None:
            df_features = fit_bycycle(sig, fs, f_range, center)

        # Restrict dataframe to frequency range
        df_osc = limit_df(df_features, fs, f_range, only_bursts=use_thresh)

        # No cycles found in frequency range
        if not isinstance(df_osc, pd.DataFrame) or len(df_osc) < min_n_cycles:
            motifs, cycles = _nan_append(motifs, cycles)
            continue

        # Split signal into 2d array of cycles
        sig_cyc = split_signal(df_osc, sig, True, center)

        # Cluster cycles
        labels = cluster_cycles(sig_cyc, min_clust_score=min_clust_score, min_clusters=min_clusters,
                                max_clusters=max_clusters, random_state=random_state)

        # Single clusters found
        if not isinstance(labels, np.ndarray):

            motif = np.mean(sig_cyc, axis=0)

            # The variance of the motif is too small (i.e. flat line)
            if np.var(motif) < var_thresh:
                motifs, cycles = _nan_append(motifs, cycles)
                continue

            motifs.append([motif])

        # Multiple motifs found at the current frequency range
        else:

            multi_motifs = []
            for idx in range(max(labels)+1):

                label_idxs = np.where(labels == idx)[0]

                motif = np.mean(sig_cyc[label_idxs], axis=0)

                if np.var(motif) < var_thresh:
                    multi_motifs.append(np.nan)
                else:
                    multi_motifs.append(motif)

            # Variance too small
            if len(multi_motifs) == 0:
                motifs, cycles = _nan_append(motifs, cycles)
                continue

            motifs.append(multi_motifs)

        # Collect cycles
        cycles['sigs'].append(sig_cyc)
        cycles['dfs_features'].append(df_osc)
        cycles['labels'].append(labels)
        cycles['f_ranges'].append(f_range)

    return motifs, cycles


def _nan_append(motifs, cycles):
    """Append nans for subthreshold motifs"""

    motifs.append(np.nan)
    for key in cycles:
        cycles[key].append(np.nan)

    return motifs, cycles
