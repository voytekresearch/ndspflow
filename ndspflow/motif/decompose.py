"""Decompose signal into periodic and aperiodic components."""


import numpy as np
from scipy.signal import resample

from neurodsp.utils.norm import normalize_sig

from ndspflow.motif.utils import motif_to_cycle


def decompose(sig, motifs, dfs_osc, center='peak', labels=None, mean_center=True):
    """Decompose a signal into its periodic/aperioidic components.

    Parameters
    ----------
    sig : 1d array
        Time series.
    motifs : list of 1d arrays
         Motifs for each center frequency
    dfs_osc : list of pd.DataFrame
        Bycycle dataframes that correspond, in order, to each motif.
    center : str, optional, {'peak', 'trough'}
        Center extrema definition.
    labels : list, optional, default: None
        Cluster labels found using :func:`~.cluster_cycles`.
    mean_center  : bool, optional, default: True
        Global detrending (mean centering of the original signal).

    Returns
    -------
    sig_pe : 1d array
        The reconstructed periodic signal.
    sig_ap : 1d array
        The reconstructed aperiodic signal.
    tfroms : 2d array, optional
        The affine matrix. Only returned when transform is 'affine'.
    """

    side = 'trough' if center == 'peak' else 'peak'

    # Intialize array of nans
    sig_ap = np.zeros_like(sig)
    sig_ap[:] = np.nan

    first_cyc_start = []
    last_cyc_end = []

    tforms = []

    for idx_motif, (motif, df_osc) in enumerate(zip(motifs, dfs_osc)):

        if not isinstance(motif, float):


            motif_tforms = []
            for idx_cyc, (_, cyc) in enumerate(df_osc.iterrows()):

                # Isolate each cycle
                start = int(cyc['sample_last_' + side])
                end = int(cyc['sample_next_' + side]) + 1
                sig_cyc = sig[start:end]

                # Resample motif if needed
                motif_idx = int(labels[idx_motif][idx_cyc]) if labels and \
                    not isinstance(labels[idx_motif], float) else 0

                if len(motif[0]) != len(sig_cyc):
                    sig_motif = resample(motif[motif_idx], len(sig_cyc))
                else:
                    sig_motif = motif[motif_idx]

                # Affine transform
                sig_motif, tform = motif_to_cycle(sig_motif, sig_cyc)
                motif_tforms.append(tform)

                # Mean center
                if mean_center:
                    sig_motif = normalize_sig(sig_motif, mean=np.mean(sig))

                # Remove motif to get aperiodic signal
                sig_ap[start:end] = (sig_cyc - sig_motif)

            # Samples of where cyclepoints start/begin
            first_cyc_start.append(df_osc['sample_last_' + side].values[0])
            last_cyc_end.append(df_osc['sample_next_' + side].values[-1])

        tforms.append(motif_tforms)

    # Fill non-burst cycles
    idxs = np.where(np.isnan(sig_ap))[0]
    sig_ap[idxs] = sig[idxs]

    # Get the periodic signal
    sig_pe = sig - sig_ap

    return sig_pe, sig_ap, tforms
