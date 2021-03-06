"""Test creation of html reports from template."""

import os
from tempfile import TemporaryDirectory
from copy import deepcopy

from ndspflow.reports.html import (generate_report, generate_header,
                                   generate_fooof_report, generate_bycycle_report)
from ndspflow.io.save import save_fooof, save_bycycle


def test_generate_report(fooof_outs, bycycle_outs, test_data):

    # Prepare bycycle args
    bm = bycycle_outs['bg']
    sig = test_data['sig_2d']
    fs = test_data['fs']
    threshold_kwargs = bycycle_outs['threshold_kwargs']
    fit_kwargs = dict(sig=sig, fs=fs, threshold_kwargs=threshold_kwargs)

    for model in [fooof_outs['fg'], fooof_outs['fgs']]:

        temp_dir = TemporaryDirectory()
        output_dir = temp_dir.name

        # Save models
        save_fooof(model, output_dir)
        save_bycycle(bm, output_dir)

        # Create report
        fname = 'report_group.html'
        generate_report(output_dir, fms=model, bms=(bm, deepcopy(fit_kwargs)), group_fname=fname)

        assert os.path.isfile(os.path.join(output_dir, 'fooof', fname))
        os.listdir(os.path.join(output_dir, 'fooof'))
        temp_dir.cleanup()


def test_generate_header():

    n_fooofs = 1
    n_bycycles = 0

    html_header = generate_header('group', '/tmp', 'fooof', n_fooofs=n_fooofs,
                                  n_bycycles=n_bycycles)

    # Assert masthead template was used
    assert "masthead" in html_header

    # Assert subject template replacement was successful
    assert "{% SUBJECT_TEMPLATE %}" not in html_header
    assert str(n_fooofs) in html_header
    assert str(n_bycycles) in html_header


def test_generate_fooof_report(fooof_outs):

    # Load data from fixture
    fm, fooof_graph = fooof_outs['fm'], fooof_outs['fm_graph']

    html_report = generate_header('subject', '/tmp', 'fooof', n_fooofs=1, n_bycycles=0)
    html_report = generate_fooof_report(fm, fooof_graph, html_report)

    # Assert html template replacement was successful
    assert "{% settings %}" not in html_report
    assert "{% model_type %}" not in html_report
    assert "{% results %}" not in html_report
    assert "{% graph %}" not in html_report


def test_generate_bycycle_report(bycycle_outs, sim_sig):

    # Load data from fixture
    sig = sim_sig['sig']
    fs = sim_sig['fs']
    threshold_kwargs = bycycle_outs['threshold_kwargs']
    graph = bycycle_outs['bm_graph']

    fit_kwargs = dict(sig=sig, fs=fs, threshold_kwargs=threshold_kwargs)

    html_report = generate_header('subject', '/tmp', 'bycycle', n_fooofs=0, n_bycycles=1)
    html_report = generate_bycycle_report(fit_kwargs, graph, html_report)

    # Assert html template replacement was successful
    assert "{% settings %}" not in html_report
    assert "{% model_type %}" not in html_report
    assert "{% results %}" not in html_report
    assert "{% graph %}" not in html_report
