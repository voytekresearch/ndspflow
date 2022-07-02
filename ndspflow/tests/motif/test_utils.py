""" Tests for motif utilitiy functions."""

from pytest import mark

import numpy as np
import pandas as pd

from ndspflow.motif.utils import motif_to_cycle, limit_df


@mark.parametrize("ttype", ['euclidean', 'similarity', 'affine', 'projective', 'polynomial'])
def test_motif_to_cycle(motif_outs, ttype):

    motif_ref = motif_outs['motif_ref']
    motif_target = motif_outs['motif_target']

    motif_trans, tform = motif_to_cycle(motif_target, motif_ref, ttype=ttype)

    assert len(motif_trans) == len(motif_ref) == len(motif_target)
    assert tform is not None


@mark.parametrize("only_burst", [True, False])
@mark.parametrize("isnan", [True, False])
def test_limit_df(test_data, bycycle_outs, only_burst, isnan):

    fs = test_data['fs']
    f_range = test_data['f_range']

    if isnan:
        f_range = (f_range[1] * 2, f_range[1] * 3)

    df_features = bycycle_outs['bm']
    df_filt = limit_df(df_features, fs, f_range, only_bursts=only_burst)

    if isnan:
        assert np.isnan(df_filt)
    else:
        assert isinstance(df_filt, pd.DataFrame)
        for freq in df_filt['freqs']:
            assert freq >= f_range[0]
            assert freq <= f_range[1]