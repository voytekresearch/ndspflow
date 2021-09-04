"""Test for spectral refitting functions."""

import pytest

import numpy as np

from ndspflow.optimize.refit import (refit, select_modes, guess_params,
    fit_gaussians, refit_aperiodic)


@pytest.mark.parametrize("power_thresh", [.1, np.inf])
@pytest.mark.parametrize("energy_thresh", [0, 2, np.inf])
@pytest.mark.parametrize("refit_ap", [True, False])
def test_refit(fooof_outs, test_data, power_thresh, energy_thresh, refit_ap):

    fm = fooof_outs['fm'].copy()

    sig = test_data['sig_1d']
    fs = test_data['fs']
    f_range = test_data['f_range']

    fm_refit, imf, pe_mask = refit(fm.copy(), sig, fs, f_range, power_thresh=power_thresh,
                                   energy_thresh=energy_thresh, refit_ap=refit_ap)

    if power_thresh < 1 and (energy_thresh == 2 or energy_thresh == 0):

        assert np.not_equal(fm_refit.peak_params_, fm.peak_params_).any()

        if refit_ap is True:
            assert np.not_equal(fm_refit.aperiodic_params_, fm.aperiodic_params_).any()
        else:
            assert np.equal(fm_refit.aperiodic_params_, fm.aperiodic_params_).all()

        assert pe_mask.any()

    else:

        assert np.equal(fm_refit.peak_params_, fm.peak_params_).all()
        assert np.equal(fm_refit.aperiodic_params_, fm.aperiodic_params_).all()
        assert not pe_mask.any()

    assert isinstance(imf, np.ndarray)
    assert isinstance(pe_mask, np.ndarray)

    assert imf.shape[0] == len(pe_mask)
    assert imf.shape[1] == len(sig)


def test_select_modes():

    powers_imf = np.zeros(10)
    powers_ap = np.zeros(10)

    pe_mask = select_modes(powers_imf.copy() + .3, powers_ap, power_thresh=0.2)
    assert pe_mask.all()

    pe_mask = select_modes(powers_imf.copy() + .3, powers_ap, power_thresh=0.4)
    assert not pe_mask.any()

    pe_mask = select_modes(powers_imf.copy() + .1, powers_ap, power_thresh=0.2)
    assert not pe_mask.any()


def test_guess_params(fooof_outs):

    fm = fooof_outs['fm']

    freqs = fm.freqs.copy()
    powers = fm.power_spectrum.copy()
    ap_fit = fm._ap_fit.copy()

    cf = fm.get_params('peak', 'CF')
    bw = fm.get_params('peak', 'BW')

    lower = np.argmin(np.abs(freqs - (cf - (2 * bw))))
    upper = np.argmin(np.abs(freqs - (cf + (2 * bw))))

    inds = np.arange(lower, upper, dtype=int)

    power_imf = np.zeros_like(ap_fit)
    power_imf[inds] = powers[inds] + 20

    guess, bounds = guess_params(freqs, powers, power_imf, ap_fit, inds)

    assert guess[0] >= freqs[lower] and guess[0] <= freqs[upper]
    assert guess[1] >= np.min(ap_fit[inds]) and guess[1] <= (np.max(powers[inds]) + 10)
    assert guess[2] >=  (.1 * (upper - lower)) and guess[2] <= (2 * (upper - lower))

    for lb, ub, g in zip(bounds[0], bounds[1], guess):
        assert lb < ub
        assert g >= lb and g <= ub


@pytest.mark.parametrize("limits", [None, True])
def test_fit_gaussians(fooof_outs, limits):

    fm = fooof_outs['fm']

    freqs = fm.freqs.copy()
    powers = fm.power_spectrum.copy()
    ap_fit = fm._ap_fit.copy()

    cf = fm.get_params('peak', 'CF')
    bw = fm.get_params('peak', 'BW')

    lower = np.argmin(np.abs(freqs - (cf - (2 * bw))))
    upper = np.argmin(np.abs(freqs - (cf + (2 * bw))))

    inds = np.arange(lower, upper, dtype=int)

    power_imf = np.zeros_like(ap_fit)
    power_imf[inds] = powers[inds] + 20

    if limits:
        limits = [[lower], [upper]]

    gauss_params = fit_gaussians(freqs, powers, power_imf, ap_fit, inds, limits=limits)

    assert gauss_params.ndim == 2
    assert gauss_params.shape[-1] == 3

    gauss_params = fit_gaussians(freqs, powers, power_imf, ap_fit, inds, limits=[[0], [1]])
    assert gauss_params is None


def test_refit_aperiodic(fooof_outs):

    fm = fooof_outs['fm']

    freqs = fm.freqs.copy()
    powers = fm.power_spectrum.copy()
    peak_fit = fm._peak_fit.copy()

    ap_params, ap_fit = refit_aperiodic(freqs, powers, peak_fit)

    assert (ap_fit == fm._ap_fit).all()
    assert len(ap_params) == 2
