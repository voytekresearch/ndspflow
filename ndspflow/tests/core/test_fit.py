"""Test FOOOOF and Bycycle fitting functions."""

import pytest
import pandas as pd

from fooof import FOOOF, FOOOFGroup
from ndspflow.core.fit import fit_fooof, fit_bycycle


@pytest.mark.parametrize("ndim", [1, 2, 3, pytest.param(4, marks=pytest.mark.xfail)])
def test_fit_fooof(ndim, test_data):

    # Load data from fixture
    freqs = test_data['freqs']

    if ndim == 1:
        powers = test_data['powers_1d']
    elif ndim == 2:
        powers = test_data['powers_2d']
    elif ndim == 3:
        powers = test_data['powers_3d']
    elif ndim == 4:
        powers = test_data['powers_4d']

    # Fit data
    model = fit_fooof(freqs, powers, freq_range=(1, 40), init_kwargs={'verbose': False}, n_jobs=1)

    if ndim == 1:

        # Check type
        assert isinstance(model, FOOOF)

        # Check that results exist
        assert model.has_model

    elif ndim == 2:

        # Check type
        assert isinstance(model, FOOOFGroup)

        # Check number of results
        assert len(model) == 2

        # Check that results exist
        for idx in range(len(model)):
            fm = model.get_fooof(idx)
            assert fm.has_model

    elif ndim == 3:

        # Check type
        assert isinstance(model, list)

        # Check number of results
        assert len(model) == 2
        for fg in model:
            assert len(fg) == 2

        # Check that results exist
        for fg in model:
            for fm_idx in range(len(fg)):
                assert fg.get_fooof(fm_idx).has_model


@pytest.mark.parametrize("ndim", [1, 2, 3, pytest.param(4, marks=pytest.mark.xfail)])
def test_fit_bycycle(ndim, test_data):

    # Get signals from fixture
    if ndim == 1:
        sig = test_data['sig_1d']
    elif ndim == 2:
        sig = test_data['sig_2d']
    elif ndim == 3:
        sig = test_data['sig_3d']
    elif ndim == 4:
        sig = test_data['sig_4d']

    # Fit
    fs = test_data['fs']
    f_range = test_data['f_range']
    df = fit_bycycle(sig, fs, f_range)

    if ndim == 1:
        assert isinstance(df, pd.DataFrame)

    elif ndim == 2:
        assert isinstance(df[0], pd.DataFrame)

    elif ndim == 3:
        assert isinstance(df[0][0], pd.DataFrame)
