"""Test result saving functions."""

import os
from tempfile import TemporaryDirectory
import pytest

from ndspflow.io.save import save_fooof
from ndspflow.core.fit import fit_fooof


@pytest.mark.parametrize("ndim", [1, 2, 3, pytest.param(4, marks=pytest.mark.xfail)])
def test_save_fooof(ndim, test_data):

     # Load data from fixture
    freqs = test_data['freqs']

    if ndim == 1:
        powers = test_data['powers_1d']
    elif ndim == 2:
        powers = test_data['powers_2d']
    elif ndim == 3:
        powers = test_data['powers_3d']

    # Fit data
    model = fit_fooof(freqs, powers, freq_range=(1, 40), init_kwargs={'verbose': False}, n_jobs=1)

    # Save
    test_dir = TemporaryDirectory()
    output_dir = test_dir.name
    save_fooof(model, output_dir)

    for f in [os.path.join(dp, f) for dp, dn, fn in os.walk(output_dir) for f in fn]:
        assert 'report.html' in f or 'results.json' in f

    test_dir.cleanup()
