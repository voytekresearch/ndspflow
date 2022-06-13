"""Tests for EMD functions."""

import numpy as np

import pytest

from ndspflow.optimize.emd import compute_emd, compute_it_emd, limit_freqs_hht


def test_compute_emd(sim_sig):

    sig = sim_sig['sig']

    imfs = compute_emd(sig)

    assert isinstance(imfs, np.ndarray)
    assert len(imfs) > 0
    assert imfs.shape[1] == len(sig)


def test_compute_it_emd(sim_sig):

    sig = sim_sig['sig']
    fs = sim_sig['fs']

    imfs = compute_it_emd(sig, fs)

    assert isinstance(imfs, np.ndarray)
    assert len(imfs) > 0
    assert imfs.shape[1] == len(sig)


@pytest.mark.parametrize("thresh", [0, 2, np.inf])
def test_limit_freqs_hht(sim_sig, thresh):

    sig = sim_sig['sig']
    fs = sim_sig['fs']

    imfs = compute_emd(sig)

    freqs = np.arange(1, 100)

    freqs_min, freqs_max = limit_freqs_hht(imfs, freqs, fs, energy_thresh=thresh)

    if thresh == np.inf:
        assert freqs_min is None
        assert freqs_max is None
    else:
        assert (freqs_min < freqs_max).all()
