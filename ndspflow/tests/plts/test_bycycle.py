"""Test bycycle plotting functions."""

from pytest import mark, param

from ndspflow.plts.bycycle import plot_bm, plot_bg


@mark.parametrize('plot_only_result', [True, False])
def test_plot_bm(bycycle_outs, test_data, plot_only_result):

    df_features = bycycle_outs['bm']
    sig = test_data['sig_1d']
    fs = test_data['fs']
    threshold_kwargs = bycycle_outs['threshold_kwargs']

    graph = plot_bm(df_features, sig, fs, threshold_kwargs, plot_only_result=plot_only_result)

    html_contains = ['Signal', 'Burst', 'Voltage<br>(normalized)', 'Time']

    for term in html_contains:
        assert term in graph


def test_plot_bg(bycycle_outs, test_data):

    sigs = test_data['sig_2d']
    fs = test_data['fs']
    bg = bycycle_outs['bg']

    graph = plot_bg(bg, sigs, fs)

    html_contains = ['plotly-graph-div', 'recolorBursts', 'rewriteBursts']

    for term in html_contains:
        assert term in graph
