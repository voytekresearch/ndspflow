"""Tests for core utility functions."""

import pytest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from fooof import FOOOF
from ndspflow.core.utils import flatten_fms, flatten_bms, limit_df


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
def test_flatten_bms(ndim, test_data, bycycle_outs):

    # Load data from fixture
    if ndim == 1:
        model = bycycle_outs['bm']
        sigs = test_data['sig_1d']
    elif ndim == 2:
        model = bycycle_outs['bg']
        sigs = test_data['sig_2d']
    elif ndim == 3:
        model = bycycle_outs['bgs']
        sigs = test_data['sig_3d']

    output_dir = '/path/to/output'

    bms, bm_paths, sigs_2d = flatten_bms(model, output_dir, sigs=sigs)

    assert len(bms) == len(bm_paths)

    for bm, bm_path in zip(bms, bm_paths):

        assert isinstance(bm, pd.DataFrame)
        assert not bm.empty

        if len(bm_paths) > 1:
            assert output_dir + '/signal' in bm_path
        else:
            assert output_dir in bm_path

    assert sigs_2d.ndim == 2


@pytest.mark.parametrize("only_burst", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("isnan", [True, False])
def test_limit_df(test_data, bycycle_outs, only_burst, verbose, isnan):

    fs = test_data['fs']
    f_range = test_data['f_range']

    if isnan:
        f_range = (f_range[1] * 2, f_range[1] * 3)

    df_features = bycycle_outs['bm']
    df_filt = limit_df(df_features, fs, f_range, only_bursts=only_burst, verbose=verbose)

    if isnan:
        assert np.isnan(df_filt)
    else:
        assert isinstance(df_filt, pd.DataFrame)
        for freq in df_filt['freqs']:
            assert freq >= f_range[0]
            assert freq <= f_range[1]

