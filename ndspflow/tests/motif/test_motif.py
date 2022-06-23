"""Test for extracting motifs."""

import numpy as np
import pandas as pd

from ndspflow.motif.fit import fit_bycycle
from ndspflow.motif import extract


def test_extract(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    # Model input
    motifs, cycles = extract(fm, sig, fs)
    _check_results(motifs, cycles, 'valid', 1)

    # Tuple input
    params = [(fm.get_params('peak_params', 'CF'), fm.get_params('peak_params', 'BW'))]
    motifs, cycles = extract(params, sig, fs)
    _check_results(motifs, cycles, 'valid', 1)

    # Dataframe input
    f_range = (params[0][0]-params[0][1], params[0][0]+params[0][1])
    df_features = fit_bycycle(sig, fs, f_range, 'peak')
    motifs, cycles = extract(params, sig, fs, df_features=df_features)
    _check_results(motifs, cycles, 'valid', 1)

    # Force multi-motif
    motifs, cycles = extract(fm, sig, fs, min_clusters=2, min_clust_score=0,
                             var_thresh=0, min_n_cycles=2)
    _check_results(motifs, cycles, 'valid', 1)

    # Force multi-motif and subthresh variance
    motifs, cycles = extract(fm, sig, fs, min_clusters=2, min_clust_score=0,
                             var_thresh=1, min_n_cycles=2)
    _check_results(motifs, cycles, 'invalid', 1)

    # Force single motif
    motifs, cycles = extract(params, sig, fs, min_clust_score=1.1)
    _check_results(motifs, cycles, 'valid', 1)

    # Minimum cycles - no cycles will survive
    motifs, cycles = extract(params, sig, fs, min_n_cycles=np.inf)
    _check_results(motifs, cycles, 'invalid', 1)

    # Sub Variance threshold - no cycles will survive
    motifs, cycles = extract(params, sig, fs, min_n_cycles=0, var_thresh=np.inf)
    _check_results(motifs, cycles, 'invalid', 1)

    # Requires lower bound step
    fm.peak_params_[0][0] = 2
    fm.peak_params_[0][2] = 1.9
    motifs, cycles = extract(fm, sig, fs)
    _check_results(motifs, cycles, 'invalid', 1)


def _check_results(motifs, cycles, validity, n_params):
    """Check results using valid or invalid parameters."""

    if validity == 'valid':
        assert len(motifs) == n_params
        for key in cycles:
            assert len(cycles[key]) == n_params
            if key == 'sigs':
                assert isinstance(cycles[key][0], np.ndarray)
            elif key == 'labels':
                if isinstance(cycles[key][0], float):
                    assert np.isnan(cycles[key][0])
                else:
                    assert isinstance(cycles[key][0], np.ndarray)
            elif key == 'dfs_features':
                assert isinstance(cycles[key][0], pd.DataFrame)
            elif key == 'f_ranges':
                assert isinstance(cycles['f_ranges'], list)

    else:

        if isinstance(motifs[0], float):
            # Whole motif rejected
            assert np.isnan(motifs[0])
            for key in cycles:
                assert len(cycles[key]) == 1
                assert np.isnan(cycles[key][0])
        else:
            # Sub-variance threshold
            assert np.isnan(motifs[0][0])
            for key in cycles:
                assert len(cycles[key]) == n_params
