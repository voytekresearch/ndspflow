"""Test motif extraction functions"""

from pytest import mark

import numpy as np

from ndspflow.core.motif import extract_motifs, split_signal, cluster_motifs


@mark.parametrize("return_cycles", [True, False])
def test_extract_motifs(test_data, bycycle_outs, fooof_outs, return_cycles):

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

    if return_cycles:

        motifs, oscs = extract_motifs(_fm, bm, sig, fs, return_cycles=return_cycles, thresh=0.1)

        for key in oscs:
            assert len(motifs) == len(oscs[key])

    else:

        motifs = extract_motifs(_fm, bm, sig, fs, return_cycles=return_cycles, thresh=0.1)

    assert len(motifs) == len(peak_params)
    assert isinstance(motifs[0][0], np.ndarray)
    assert np.isnan(motifs[1])



def test_split_signal(test_data, bycycle_outs):

    df_osc = bycycle_outs['bm'].copy()
    df_osc = df_osc[df_osc['is_burst'] == True]

    sig = test_data['sig_1d']
    center = 'peak'

    sig_cyc = split_signal(df_osc, sig, True, center)

    assert sig_cyc.ndim == 2
    assert np.shape(sig_cyc)[0] == len(df_osc)
    assert np.shape(sig_cyc)[1] == np.mean(df_osc['period'].values, dtype=int)


def test_cluster_motifs(test_data, bycycle_outs):

    df_osc = bycycle_outs['bm'].copy()
    df_osc = df_osc[df_osc['is_burst'] == True]

    sig = test_data['sig_1d']
    center = 'peak'

    motifs = split_signal(df_osc, sig, True, center)

    labels = cluster_motifs(motifs, thresh=0.1, max_clusters=10)
    assert len(labels) == len(df_osc)

    osc = split_signal(df_osc, sig, True, center)[0]
    motifs = np.reshape(osc, (1, len(osc)))

    labels = cluster_motifs(motifs, thresh=0.1, max_clusters=10)
    assert np.isnan(labels)
