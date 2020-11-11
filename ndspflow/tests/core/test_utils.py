"""Tests for core utility functions."""

import pytest
from tempfile import TemporaryDirectory

import pandas as pd

from fooof import FOOOF
from ndspflow.core.utils import flatten_fms, flatten_bms


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_flatten_fms(ndim, fooof_outs):

    # Load data from fixture
    if ndim == 1:
        model = fooof_outs['fm']
    elif ndim == 2:
        model = fooof_outs['fg']
    elif ndim == 3:
        model = fooof_outs['fgs']

    output_dir = '/path/to/output'

    fms, fm_paths = flatten_fms(model, output_dir)

    assert len(fms) == len(fm_paths)

    for fm, fm_path in zip(fms, fm_paths):

        assert isinstance(fm, FOOOF)
        assert fm.has_model

        if len(fm_paths) > 1:
            assert output_dir + '/spectrum' in fm_path
        else:
            assert output_dir in fm_path


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_flatten_bms(ndim, bycycle_outs):

    # Load data from fixture
    if ndim == 1:
        model = bycycle_outs['bm']
    elif ndim == 2:
        model = bycycle_outs['bg']
    elif ndim == 3:
        model = bycycle_outs['bgs']

    output_dir = '/path/to/output'

    bms, bm_paths = flatten_bms(model, output_dir)

    assert len(bms) == len(bm_paths)

    for bm, bm_path in zip(bms, bm_paths):

        assert isinstance(bm, pd.DataFrame)
        assert not bm.empty

        if len(bm_paths) > 1:
            assert output_dir + '/signal' in bm_path
        else:
            assert output_dir in bm_path


