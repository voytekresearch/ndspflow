"""Test motif extraction functions"""

from pytest import mark, param

import numpy as np

from neurodsp.sim import sim_oscillation

from ndspflow.core.motif import extract_motif, extract_motifs


def test_extract_motifs(test_data, bycycle_outs, fooof_outs):


    # Get sim settings
    fs = test_data['fs']
    sig = test_data['sig_1d']

    # Get bycycle
    bm = bycycle_outs['bm']

    # Get fooof
    fm = fooof_outs['fm']

    # Insert an artificial 20hz peak
    _fm = fm.copy()
    peak_params = _fm.peak_params_.copy()
    peak_params = np.repeat(peak_params, 2, axis=0)
    peak_params[1][0] = 20
    _fm.peak_params_ = peak_params

    motifs, dfs_osc = extract_motifs(_fm, bm, sig, fs)

    assert len(motifs) == len(dfs_osc)
    assert len(motifs) == len(peak_params)
    assert isinstance(motifs[0], np.ndarray)
    assert np.isnan(motifs[1])


@mark.parametrize("normalize", [True, False])
def test_extract_motif(test_data, bycycle_outs, normalize):

    sig = test_data['sig_1d']
    fs = test_data['fs']
    df_features = bycycle_outs['bm']

    motif = extract_motif(df_features, sig, scaling=1, normalize=normalize, center='peak')

    assert isinstance(motif, np.ndarray)

    min_freq = 1 / (df_features['period'].max() / fs)
    max_freq = 1 / (df_features['period'].min() / fs)
    motif_freq = 1 / (len(motif) / fs)

    assert motif_freq >= min_freq and motif_freq <= max_freq


