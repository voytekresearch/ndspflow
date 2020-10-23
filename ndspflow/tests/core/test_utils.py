"""Tests for core utility functions."""

import pytest
from tempfile import TemporaryDirectory
from fooof import FOOOF
from ndspflow.core.utils import flatten_fms


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_flatten_fms(ndim, fm_outs, fg_outs, fgs_outs):

    # Load data from fixture

    if ndim == 1:
        model = fm_outs['fm']
    elif ndim == 2:
        model = fg_outs['fg']
    elif ndim == 3:
        model = fgs_outs['fgs']

    output_dir = '/path/to/output'

    fms, fm_paths = flatten_fms(model, output_dir)

    assert len(fms) == len(fm_paths)

    for fm, fm_path in zip(fms, fm_paths):

        assert type(fm) is FOOOF
        assert fm.has_model

        if len(fm_paths) > 1:
            assert '/path/to/output/spectrum' in fm_path
        else:
            assert '/path/to/output' in fm_path
