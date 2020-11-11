"""Test bycycle plotting functions."""

from pytest import mark, param

from ndspflow.plts.bycycle import plot_bm


@mark.parametrize('plot_only_result', [True, False])
def test_plot_bm(bycycle_outs, sim_sig, plot_only_result):

    df_features = bycycle_outs['bm']
    sig = sim_sig['sig']
    fs = sim_sig['fs']
    threshold_kwargs = bycycle_outs['threshold_kwargs']

    graph = plot_bm(df_features, sig, fs, threshold_kwargs, plot_only_result=plot_only_result)

    html_contains = ['Signal', 'Burst', 'Voltage (normalized)', 'Time']

    for term in html_contains:
        assert term in graph
