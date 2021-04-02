"""Tests for motif burst detection."""

import numpy as np


from ndspflow.motif import motif_burst_detection, extract


def test_motif_burst_detection(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    motifs, cycles = extract(fm, sig, fs, clust_score=1.1, only_bursts=False)

    is_burst = motif_burst_detection(motifs[0], cycles['dfs_features'][0], sig)

    assert len(is_burst) == len(cycles['dfs_features'][0])
    assert isinstance(is_burst, np.ndarray)
    assert is_burst.dtype == 'bool'

    # Max correlation coeff is 1, no bursts will be super threshold
    is_burst = motif_burst_detection(motifs[0], cycles['dfs_features'][0], sig, corr_thresh=2)

    assert not is_burst.any()
