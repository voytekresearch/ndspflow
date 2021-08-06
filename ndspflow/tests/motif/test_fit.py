"""Tests for the Motif class."""

from pytest import raises

import numpy as np

from ndspflow.motif import Motif, MotifGroup
from ndspflow.motif.fit import MotifResult, _motif_proxy


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
    assert isinstance(motif[0].sig_ap, np.ndarray)
    assert isinstance(motif[0].sig_pe, np.ndarray)
    assert motif[0].sig_ap.shape == motif[0].sig_pe.shape

    motif.decompose(transform=True)
    assert isinstance(motif[0].tforms, list)
    assert isinstance(motif[0].tforms[0].params, np.ndarray)
    assert motif[0].tforms[0].params.shape == (3, 3)

    # Fitting must be prior to decomposition
    motif = Motif()
    with raises(ValueError):
        motif.decompose()



def test_motif_plot(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    motif = Motif()
    motif.fit(fm, sig, fs)
    motif.plot(show=False)

def test_motif_plot_decompose(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    motif = Motif()
    motif.fit(fm, sig, fs)
    motif.plot_decompose(0)


def test_motif_plot_spectra(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    motif = Motif()
    motif.fit(fm, sig, fs)
    motif.plot_spectra(0)


def test_motif_plot_transform(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    motif = Motif()
    motif.fit(fm, sig, fs)
    motif.plot_transform(0)


def test_motif_group_fit(test_data, fooof_outs):

    fg = fooof_outs['fg']
    sigs = test_data['sig_2d']
    fs = test_data['fs']

    motif_group = MotifGroup()

    motif_group.fit(fg, sigs, fs)

    assert len(motif_group) == len(sigs)

    for motif in motif_group:
        assert isinstance(motif, Motif)

    assert isinstance(motif_group[0], Motif)


def test_motif_proxy(sim_sig, fooof_outs):

    sig = sim_sig['sig']
    fs = sim_sig['fs']
    fm = fooof_outs['fm']

    motif = _motif_proxy([sig, fm], fs, {})

    assert isinstance(motif, Motif)
