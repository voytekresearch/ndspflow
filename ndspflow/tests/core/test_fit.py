"""Test FOOOOF and Bycycle fitting functions."""

import os
import pytest
import numpy as np

from fooof import FOOOF, FOOOFGroup
from ndspflow.core.fit import fit_fooof


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
        assert type(model) is FOOOF

        # Check that results exist
        assert model.has_model

    elif ndim == 2:

        # Check type
        assert type(model) is FOOOFGroup

        # Check number of results
        assert len(model) == 2

        # Check that results exist
        for idx in range(len(model)):
            fm = model.get_fooof(idx)
            assert fm.has_model

    elif ndim == 3:

        # Check type
        assert type(model) is list

        # Check number of results
        assert len(model) == 2
        for fg in model:
            assert len(fg) == 2

        # Check that results exist
        for fg in model:
            for fm_idx in range(len(fg)):
                assert fg.get_fooof(fm_idx).has_data
