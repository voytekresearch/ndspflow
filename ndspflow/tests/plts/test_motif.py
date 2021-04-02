"""Test motif plotting functions."""

import numpy as np
import plotly.graph_objs as go

from ndspflow.motif import extract
from ndspflow.plts.motif import plot_motifs


def test_plot_motifs(fooof_outs, bycycle_outs, test_data):

    fs = test_data['fs']
    sig = test_data['sig_1d']

    fm = fooof_outs['fm']

    # Insert artificial peak so no oscillation will be found for one peak
    _fm = fm.copy()
    peak_params = _fm.peak_params_.copy()
    peak_params = np.repeat(peak_params, 2, axis=0)
    peak_params[1][0] = 20
    _fm.peak_params_ = peak_params

    df_features = bycycle_outs['bm']
    motifs, cycles = extract(_fm, sig, fs, df_features=df_features)
    fig = plot_motifs(_fm, motifs, cycles, sig, fs)
    assert isinstance(fig, go.Figure)

    html = fig.to_html(full_html=False, include_plotlyjs=False)

    terms = ['Spectrum Fit', ' hz Motif', 'Frequencies',
             'log(Power)', 'Normalized Voltage', 'Time (s)']

    for term in terms:
        assert term in html
