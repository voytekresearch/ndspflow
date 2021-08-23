"""Test for spectral refitting functions."""

import pytest

import numpy as np

from ndspflow.optimize.refit import (refit, select_modes, guess_params,
    fit_gaussians, refit_aperiodic)


@pytest.mark.parametrize("power_thresh", [.2, np.inf])
@pytest.mark.parametrize("energy_thresh", [0, 2, np.inf])
@pytest.mark.parametrize("refit_ap", [True, False])
def test_refit(fooof_outs, test_data, power_thresh, energy_thresh, refit_ap):

    fm = fooof_outs['fm']

    sig = test_data['sig_1d']
    fs = test_data['fs']
    f_range = test_data['f_range']

    fm_refit, imf, pe_mask = refit(fm.copy(), sig, fs, f_range, power_thresh=power_thresh,
                                   energy_thresh=energy_thresh, refit_ap=refit_ap)

    if power_thresh == .2 and (energy_thresh == 2 or energy_thresh == 0):

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


def test_guess_params():
    pass

def test_fit_gaussians():
    pass

def test_refit_aperiodic():
    pass
