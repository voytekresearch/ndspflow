"""Test FOOOF plotting functions."""


from pytest import mark, param
from ndspflow.plts.fooof import plot_fm, plot_fg, plot_fgs


@mark.parametrize("fill_gaussians", [True, ['red']])
def test_plot_fm(fooof_outs, fill_gaussians):

    fig = plot_fm(fooof_outs['fm'], fill_gaussians=fill_gaussians)
    graph = fig.to_html(full_html=False, include_plotlyjs=False)

    assert type(graph) is str

    html_contains = ['Original', 'Full Fit', 'Aperiodic Fit', 'Frequencies', 'log(Power)']

    for term in html_contains:
        assert term in graph

def test_plot_fg(fooof_outs):

    fg = fooof_outs['fg']
    urls = ['' for i in range(len(fg))]

    graph = plot_fg(fg, urls)

    assert type(graph) is str

    html_contains = ['Error', 'R-Squared', 'Peak Parameters', 'Number of Peaks', 'Center Frequency',
                     'Band Width', 'Peak Width', 'Aperiodic Parameters', 'Exponent', 'Offset']

    for term in html_contains:
        assert term in graph


def test_plot_fgs(fooof_outs):

    fgs = fooof_outs['fgs']
    urls = ['' for fg in fgs for fm in fg]

    graphs = plot_fgs(fgs, urls)

    assert type(graphs) is list

    for idx, graph in enumerate(graphs):

        if idx % 2 == 0:

            assert type(graph) is str
            assert 'Group Index' in graph

        else:

            html_contains = ['Error', 'R-Squared', 'Peak Parameters', 'Number of Peaks',
                             'Center Frequency', 'Band Width', 'Peak Width', 'Aperiodic Parameters',
                             'Exponent', 'Offset']

            for term in html_contains:
                assert term in graph
