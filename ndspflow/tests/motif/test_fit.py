"""Tests for the Motif class."""

import numpy as np

from ndspflow.motif import Motif
from ndspflow.motif.fit import MotifResult


def test_motif():
    """Test initializing the object"""

    corr_thresh = 0.5
    var_thresh = 0.05
    min_clust_score = 0.5
    min_clusters = 2
    max_clusters = 10
    min_n_cycles = 10
    center = 'peak'

    motif = Motif(corr_thresh, var_thresh, min_clust_score,
                  min_clusters, max_clusters, min_n_cycles, center)

    # Defaults
    assert motif.corr_thresh == corr_thresh
    assert motif.var_thresh == var_thresh
    assert motif.min_clust_score == min_clust_score
    assert motif.min_clusters == min_clusters
    assert motif.max_clusters == max_clusters
    assert motif.min_n_cycles == min_n_cycles
    assert motif.center == center

    # Fit args
    assert motif.fm == None
    assert motif.sig == None
    assert motif.fs == None

    # Results
    assert motif.results == []
    assert motif.sig_pe == None
    assert motif.sig_ap == None
    assert motif.tforms == None


def test_motif_fit(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    motif = Motif()

    motif.fit(fm, sig, fs)
    assert len(motif.results) == len(fm.get_params('peak_params'))

    # No cycles in one f_range
    params = [(fm.get_params('peak_params', 'CF'), fm.get_params('peak_params', 'BW')),
              (100, 10)]
    motif.fit(params, sig, fs)
    assert len(motif.results) == len(params)

    # No  cycles >= correlation coeff threshold
    motif = Motif(corr_thresh=1.1)
    motif.fit(fm, sig, fs, )
    assert len(motif.results) == len(fm.get_params('peak_params'))

    # Test iterating over object
    for result in motif:
        assert isinstance(result, MotifResult)

    # Test indexing object
    assert isinstance(motif[0], MotifResult)

    # Test that length is accessible
    assert len(motif) == fm.n_peaks_


def test_motif_decompose(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    motif = Motif()

    motif.fit(fm, sig, fs)

    motif.decompose(transform=False)
    assert isinstance(motif.sig_ap, np.ndarray)
    assert isinstance(motif.sig_pe, np.ndarray)
    assert motif.sig_ap.shape == motif.sig_pe.shape

    motif.decompose(transform=True)
    assert isinstance(motif.tforms, list)
    assert isinstance(motif.tforms[0], list)
    assert isinstance(motif.tforms[0][0].params, np.ndarray)
    assert motif.tforms[0][0].params.shape == (3, 3)


def test_motif_plot(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    motif = Motif()
    motif.fit(fm, sig, fs)

    motif = Motif()
    motif.fit(fm, sig, fs)
    motif.plot(show=False)

def test_motif_plot_decompose(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    motif = Motif()
    motif.fit(fm, sig, fs)
