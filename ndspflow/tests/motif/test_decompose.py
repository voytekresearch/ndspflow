"""Test decomposing a signal."""

from pytest import mark, param

import numpy as np

from ndspflow.motif import extract, decompose


def test_decompose(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    freq = sim_sig['freq']
    fm = fooof_outs['fm']

    motifs, cycles = extract(fm, sig, fs)

    sig_ap, sig_pe, tforms = decompose(sig, motifs, cycles['dfs_features'], center='peak',
                                       labels=cycles['labels'], mean_center=True, transform=True)

    assert len(sig_ap) == len(sig_pe) == len(tforms) == 1
    assert sig_ap[0].shape == sig_pe[0].shape

    assert len(tforms[0]) == len(cycles['dfs_features'][0])
    assert all([tform.params.shape == (3, 3) for tform in tforms[0]])

    # Don't use affine transform
    sig_ap, sig_pe = decompose(sig, motifs, cycles['dfs_features'], center='peak',
                               labels=cycles['labels'], mean_center=True, transform=False)

    # Periodic signal will have greater variation
    assert np.var(sig_ap[0]) < np.var(sig_pe[0])
    assert len(sig_ap) == len(sig_pe) == len(tforms) == 1
    assert sig_ap[0].shape == sig_pe[0].shape
