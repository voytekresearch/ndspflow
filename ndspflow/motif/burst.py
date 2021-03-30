"""Motif burst detection."""

import numpy as np
from scipy.signal import resample


def motif_burst_detection(motifs, df_features, sig, corr_thresh=5):
    """Use motifs to detect bursts.

    Parameters
    ----------
    motifs : list of 1d arrays
        A list of motifs at a single frequency. Single index of `motifs` returned from
        :func:`~.ndspflow.motif.extract`.
    df_features : pd.DataFrame
        Bycycle features fit to the corresponding motif(s) frequency range.
    sig : 1d array
        Voltage time series.
    corr_thresh : float, optional, default: 5
        Cross-correlation coefficient threshold.

    Returns
    -------
    is_burst : 1d array
        Boolean array where True marks where the signal is bursting.
    """

    is_burst = np.zeros((len(motifs), len(df_features)), dtype='bool')

    for idx, motif in enumerate(motifs):

        for row, (_, cyc) in enumerate(df_features.iterrows()):

            # Slice, normalize, and resample each cycle
            cyc = sig[cyc['sample_last_trough']:cyc['sample_next_trough']]
            cyc = resample(cyc, len(motif))

            # Correlation coefficient
            coeff = np.correlate(cyc, motif, mode='valid')[0]

            if coeff >= corr_thresh:
                is_burst[idx][row] = True

    is_burst = np.sum(is_burst, axis=0, dtype=bool)

    return is_burst
