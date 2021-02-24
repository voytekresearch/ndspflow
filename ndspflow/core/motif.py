"""Extract motifs using fooof and bycycle."""

import warnings

import numpy as np
from scipy.signal import resample
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from neurodsp.utils.norm import normalize_sig
from ndspflow.core.utils import limit_df


def extract_motifs(fm, df_features, sig, fs, scaling=1, only_bursts=True,
                   center='peak', thresh=1, max_clusters=10, return_cycles=False):
    """Get the average cycle from a bycycle dataframe for all fooof peaks.

    Parameters
    ----------
    fm : fooof.FOOOF, optional, or list of tuple
        A fooof model that has been fit, or a list of (center_freq, bandwidth).
    df_features : pandas.DataFrame
        A dataframe containing bycycle features.
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    scaling : float, optional, default: 1
        The scaling of the bandwidth from the center frequencies to limit cycles to.
    only_burst : bool, optional, default: True
        Limits the dataframe to bursting cycles when True.
    center : {'peak', 'trough'}, optional
        The center definition of cycles.
    thresh : float, optional, default: 1
        The silhouette score for accepting putative k clusters.
    max_clusters : int, optional, default: 10
        The maximum number of clusters to evaluate.
    return_cycles : bool, optiona, default: False
        Returns the signal, dataframe, and label for each cycle.

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
        labels = cluster_motifs(sig_cyc, thresh=thresh, max_clusters=max_clusters)

        # Collect cycles if requested
        cycles['sigs'].append(sig_cyc)
        cycles['dfs_osc'].append(df_osc)
        cycles['labels'].append(labels)
        cycles['f_ranges'].append(f_range)

        if not isinstance(labels, np.ndarray):
            # No superthreshold clusters found
            motifs.append([np.mean(sig_cyc, axis=0)])
        else:
            # Multiple motifs found at the current frequency range
            multi_motifs = []
            for idx in range(max(labels)+1):
                multi_motifs.append(np.mean(sig_cyc[np.where(labels == idx)[0]], axis=0))
            motifs.append(multi_motifs)

    if return_cycles:
        return motifs, cycles

    return motifs


def split_signal(df_osc, sig, normalize=True, center='peak'):
    """Split the signal from a bycycle dataframe.

    Parameters
    ----------
    df_osc : pandas.DataFrame
        A dataframe containing bycycle features, that has been limited to an oscillation frequency
        range of interest.
    sig : 1d array
        Time series.
    normalize : bool, optional, default: True
        Normalizes each cycle (mean centers with variance of one) when True.
    center : {'peak', 'trough'}, optional
        The center definition of cycles.

    Returns
    -------
    sigs : 2d array
        Each cycle resampled to the mean of the dataframe.
    """

    # Get the start/end samples of each cycle
    side = 'trough' if center == 'peak' else 'peak'
    cyc_start = df_osc['sample_last_' + side].values
    cyc_end = df_osc['sample_next_' + side].values

    # Get the average number of samples
    n_samples = np.mean(df_osc['period'].values, dtype=int)

    sigs = np.zeros((len(df_osc), n_samples))

    # Slice cycles and resample to center frequency
    for idx, (start, end) in enumerate(zip(cyc_start, cyc_end)):

        sig_cyc = sig[start:end]
        sig_cyc = resample(sig_cyc, num=n_samples)

        if normalize:
            sig_cyc = normalize_sig(sig_cyc, mean=0)

        sigs[idx] = sig_cyc

    return sigs


def cluster_motifs(motifs, thresh=0.5, min_clusters=2, max_clusters=10):
    """K-means clustering of motifs.

    Parameters
    ----------
    motifs : 2D array
        Cycles within a frequency range.
    thresh : float, optional, default: 0.5
        The silhouette score for accepting putative k clusters.
    max_cluster : int, optional, default: 10
        The maximum number of clusters to evaluate.

    Returns
    -------
    labels : 1d array
        The predicted cluster each motif belongs to.
    """

    # Nothing to cluster
    if len(motifs) == 1 or max_clusters == 1:
        return np.nan

    max_clusters = len(motifs) if len(motifs) < max_clusters else max_clusters

    labels = []
    scores = []
    for n_clusters in range(min_clusters, max_clusters+1):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clusters = KMeans(n_clusters=n_clusters, algorithm="full").fit_predict(motifs)

        labels.append(clusters)

        scores.append(silhouette_score(motifs, clusters))

    # No superthreshold clusters found
    if max(scores) < thresh:
        return np.nan

    # Split motifs based on highest silhouette score
    labels = labels[np.argmax(scores)]

    return labels


def decompose_sig(sig, motifs, dfs_osc, center='peak', non_bursts='nan'):
    """Decompose a signal into its periodic/aperioidic components.

    Parameters
    ----------
    sig : 1d array
        Time series.
    motifs : list of 1d arrays
         Motifs for each center frequency

    sig_ap_re, sig_pe_re
    """

    side = 'trough' if center == 'peak' else 'peak'


    # Intialize array of nans
    sig_ap_re = np.zeros_like(sig)
    sig_ap_re[:] = np.nan

    first_cyc_start = []
    last_cyc_end = []

    for motif, df_osc in zip(motifs, dfs_osc):

        if not isinstance(motif, float):

            for _, cyc in df_osc.iterrows():

                # Isolate each cycle
                start = int(cyc['sample_last_' + side])
                end = int(cyc['sample_next_' + side]) + 1
                sig_cyc = sig[start:end]

                # Resample motif if needed
                if len(motif[0]) != len(sig_cyc):
                    sig_motif = resample(motif[0], len(sig_cyc))
                else:
                    sig_motif = motif[0]

                # Remove motif to get aperiodic signal
                sig_ap_re[start:end] = (sig_cyc - sig_motif)

            # Samples of where cyclepoints start/begin
            first_cyc_start.append(df_osc['sample_last_' + side].values[0])
            last_cyc_end.append(df_osc['sample_next_' + side].values[-1])

    # Fill non-burst cycles
    if non_bursts == 'nan':

        first_cyc_start = int(min(first_cyc_start))
        last_cyc_end = int(max(last_cyc_end)) + 1

        sig_ap_re[:first_cyc_start] = np.nan
        sig_ap_re[last_cyc_end:] = np.nan

    elif non_bursts == 'aperiodic':

        idxs = np.where(np.isnan(sig_ap_re))[0]
        sig_ap_re[idxs] = sig[idxs]

    # Get the periodic signal
    sig_pe_re = sig - sig_ap_re

    return sig_ap_re, sig_pe_re
