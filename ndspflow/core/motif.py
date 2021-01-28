"""Extract motifs using fooof and bycycle."""

import numpy as np
from scipy.signal import resample
import pandas as pd

from neurodsp.utils.norm import normalize_sig
from ndspflow.core.utils import limit_df


def extract_motifs(fm, df_features, sig, fs, scaling=1, normalize=True,
                   weights=None, only_bursts=True, center='peak'):
    """Get the average cycle from a bycycle dataframe for all fooof peaks.

    Parameters
    ----------
    fm : fooof.FOOOF
        A fooof model that has been fit.
    df_features : pandas.DataFrame
        A dataframe containing bycycle features.
    sig : 1d array
        Time series.
    scaling : float, optional, default: 1
        The scaling of the bandwidth from the center frequencies to limit cycles to.
    normalize : bool, optional, default: True
        Normalizes each cycle (mean centers with variance of one) when True.
    weights : 1d array, optional, default: None
        Used for weighting cycles (i.e. a function of distance from the center frequency), when
        averaging cycle waveforms.
    only_burst : bool, optional, default: True
        Limits the dataframe to bursting cycles when True.
    center : {'peak', 'trough'}, optional
        The center definition of cycles.

    Returns
    -------
    motifs : list of 1d arrays
        Motifs for each center frequency in ascending order.
    dfs_osc : list of pd.DataFrame
        The subsetted dataframe used to find each motif.
    """

    # Extract center freqs and bandwidths from fooof fit
    cfs = fm.get_params('peak_params', 'CF')
    bws = fm.get_params('peak_params', 'BW')

    cfs = cfs if isinstance(cfs, (list, np.ndarray)) else [cfs]
    bws = bws if isinstance(bws, (list, np.ndarray)) else [bws]

    f_ranges = [(cf-(scaling * bws[idx]), cf+(scaling * bws[idx])) for idx, cf in enumerate(cfs)]

    # Get cycles within freq ranges
    motifs = []
    dfs_osc = []

    for f_range in f_ranges:

        # Restrict dataframe to frequency range
        df_osc = limit_df(df_features, fs, f_range, only_bursts=only_bursts)
        dfs_osc.append(df_osc)

        # No cycles left after limiting
        if not isinstance(df_osc, pd.DataFrame):
            motifs.append(np.nan)
            continue

        motif = extract_motif(df_osc, sig, normalize, weights, center)
        motifs.append(motif)

    return motifs, dfs_osc


def extract_motif(df_osc, sig, normalize=True, weights=None, center='peak'):
    """Get the average cycle from a bycycle dataframe.

    Parameters
    ----------
    df_osc : pandas.DataFrame
        A dataframe containing bycycle features, that has been limited to an oscillation frequency
        range of intereset.
    sig : 1d array
        Time series.
    normalize : bool, optional, default: True
        Normalizes each cycle (mean centers with variance of one) when True.
    weights : 1d array, optional, default: None
        Used for weighting cycles (i.e. a function of distance from the center frequency), when
        averaging cycle waveforms.
    center : {'peak', 'trough'}, optional
        The center definition of cycles.

    Returns
    -------
    sig_motif : 1d array
        The averaged waveform.

    Notes
    -----

    - The returned motif will contain a number of samples equal to the mean in the dataframe.
    - When no cycles are found within the frequency range, np.nan is returned.

    """

    # Get the start/end samples of each cycle
    side = 'trough' if center == 'peak' else 'peak'
    cyc_start = df_osc['sample_last_' + side].values
    cyc_end = df_osc['sample_next_' + side].values

    # Get the average number of samples
    samples = np.array([end-start for start, end in zip(cyc_start, cyc_end)])

    n_samples = np.mean(samples, dtype=int)

    sig_motif = np.zeros((len(cyc_start), n_samples))

    # Slice cycles and resample to center frequency
    for idx, (start, end) in enumerate(zip(cyc_start, cyc_end)):

        sig_cyc = sig[start:end]
        sig_cyc = resample(sig_cyc, num=n_samples)

        if normalize:
            sig_cyc = normalize_sig(sig_cyc, mean=0, variance=1)

        sig_motif[idx] = sig_cyc

    # Take the weigthed average of the cycles
    sig_motif = np.average(sig_motif, axis=0, weights=weights)

    return sig_motif
