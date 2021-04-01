"""Motif burst detection."""

import numpy as np
from scipy.signal import resample

from ndspflow.motif.utils import motif_to_cycle


def motif_burst_detection(motifs, df_features, sig, corr_thresh=.75):
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
        Correlation coefficient threshold.

    Returns
    -------
    is_burst : 1d array
        Boolean array where True marks where the signal is bursting.
    """

    is_burst = np.zeros((len(motifs), len(df_features)), dtype='bool')

    for idx, motif in enumerate(motifs):

        for row, (_, cyc) in enumerate(df_features.iterrows()):

            # Slice, normalize, and resample
            cyc = sig[cyc['sample_last_trough']:cyc['sample_next_trough']]

            motif_resamp = resample(motif, len(cyc))
            motif_tform, _ = motif_to_cycle(motif_resamp, cyc)

            # Correlation coefficient
            coeff_cyc_tform = np.corrcoef(cyc, motif_tform)[0][1]
            coeff_resamp_tform = np.corrcoef(motif_resamp, motif_tform)[0][1]

            if coeff_cyc_tform >= corr_thresh and coeff_resamp_tform >= corr_thresh:
                is_burst[idx][row] = True

    is_burst = np.sum(is_burst, axis=0, dtype=bool)

    return is_burst
