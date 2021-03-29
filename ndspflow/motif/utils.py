"""Utitlity array functions for determining motifs."""


import numpy as np
from scipy.signal import resample

from neurodsp.utils.norm import normalize_sig


def split_signal(df_osc, sig, normalize=True, center='peak'):
    """Split the signal using a bycycle dataframe.

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

        if start < 0:
            continue

        sig_cyc = sig[start:end]
        sig_cyc = resample(sig_cyc, num=n_samples)

        if normalize:
            sig_cyc = normalize_sig(sig_cyc, mean=0)

        sigs[idx] = sig_cyc

    return sigs




