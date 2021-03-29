"""Decompose signal into periodic and aperiodic components."""


import numpy as np
from scipy.signal import resample

def decompose_sig(sig, motifs, dfs_osc, center='peak', non_bursts='nan', labels=None):
    """Decompose a signal into its periodic/aperioidic components.

    Parameters
    ----------
    sig : 1d array
        Time series.
    motifs : list of 1d arrays
         Motifs for each center frequency
    """

    side = 'trough' if center == 'peak' else 'peak'

    # Intialize array of nans
    sig_ap_re = np.zeros_like(sig)
    sig_ap_re[:] = np.nan

    first_cyc_start = []
    last_cyc_end = []

    for idx_motif, (motif, df_osc) in enumerate(zip(motifs, dfs_osc)):

        if not isinstance(motif, float):

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
