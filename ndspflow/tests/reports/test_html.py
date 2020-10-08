"""Test creation of html reports from template."""

import os
from tempfile import TemporaryDirectory
import pytest
import numpy as np

from ndspflow.reports.html import *
from ndspflow.io.save import save_fooof
from ndspflow.tests.settings import TEST_DATA_PATH


def test_generate_report(fg_outs, fgs_outs):

    for model in [fg_outs['fg'], fgs_outs['fgs']]:

        temp_dir = TemporaryDirectory()
        output_dir = temp_dir.name
        save_fooof(model, output_dir)

        fname = 'report_group.html'
        generate_report(output_dir, fms=model, bms=None, group_fname=fname)

        assert os.path.isfile(os.path.join(output_dir, 'fooof', fname))
        temp_dir.cleanup()


def test_generate_3d_report(fgs_outs):

        # Load data from fixture
        fgs, fooof_graph = fgs_outs['fgs'], fgs_outs['fooof_graph']

        # Define output
        out_dir = TEST_DATA_PATH
        fname = 'report_group.html'

        # Embed plots
        generate_3d_report(fgs, fooof_graph, int(len(fgs)*len(fgs[0])), 0, out_dir, fname)

        # Assert that the html file was generated
        assert os.path.isfile(os.path.join(out_dir, fname))

        # Cleanup
        os.remove(os.path.join(out_dir, fname))


def test_generate_2d_report(fg_outs):

        # Load data from fixture
        fg, fooof_graph = fg_outs['fg'], fg_outs['fooof_graph']

        # Define output
        out_dir = TEST_DATA_PATH
        fname = 'report_group.html'

        # Embed plots
        generate_2d_report(fg, fooof_graph, len(fg), 0, out_dir, fname=fname)

        # Assert that the html file was generated
        assert os.path.isfile(os.path.join(out_dir, fname))

        # Cleanup
        os.remove(os.path.join(out_dir, fname))


def test_generate_1d_report(fm_outs):

        # Load data from fixture
        fm, fooof_graph = fm_outs['fm'], fm_outs['fooof_graph']

        # Define output
        out_dir = TEST_DATA_PATH
        fname = 'ndspflow.html'

        # Embed plots
        fname = 'report_test.html'
        generate_1d_report(fm, 'spectrum_0000', fooof_graph, out_dir, fname)

        # Assert that the html file was generated
        assert os.path.isfile(os.path.join(out_dir, fname))

        # Cleanup
        os.remove(os.path.join(out_dir, fname))


def test_generate_header():

    n_fooofs = 1
    n_bycycles = 0

    html_header = generate_header('subject', n_fooofs=n_fooofs, n_bycycles=n_bycycles)

    # Assert masthead template was used
    assert "masthead" in html_header

    # Assert subject template replacement was successful
    assert "{% SUBJECT_TEMPLATE %}" not in html_header

    assert str(n_fooofs) in html_header
    assert str(n_bycycles) in html_header


def test_generate_fooof_report(fm_outs):

    # Load data from fixture
    fm, fooof_graph = fm_outs['fm'], fm_outs['fooof_graph']

    html_report = generate_header('subject', n_fooofs=1, n_bycycles=0)
    html_report = generate_fooof_report(fm, fooof_graph, html_report)

    # Assert html template replacement was successful
    assert "{% fooof_settings %}" not in html_report
    assert "{% fooof_results %}" not in html_report
    assert "{% fooof_graph %}" not in html_report
