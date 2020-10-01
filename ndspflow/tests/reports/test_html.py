"""Test creation of html reports from template."""

import os
import pytest
import numpy as np

from fooof import FOOOF
from ndspflow.plts.fooof import plot_fooof
from ndspflow.reports.html import generate_1d_report, generate_header, generate_fooof_report
from ndspflow.tests.settings import TEST_DATA_PATH


@pytest.fixture(scope='session')
def fooof_outs(input_dir=TEST_DATA_PATH):

    # Load data
    freqs = np.load(os.path.join(input_dir, 'freqs.npy'))
    spectrum = np.load(os.path.join(input_dir, 'spectrum.npy'))
    freq_range = (1, 40)

    # Fit
    fm = FOOOF(peak_width_limits=(0.5, 12.0), max_n_peaks=np.inf, min_peak_height=0.0,
               peak_threshold=2.0, aperiodic_mode='fixed', verbose=False)

    fm.fit(freqs, spectrum, freq_range)

    # Plot
    fooof_graph = plot_fooof(fm)

    return [fm, fooof_graph]


def test_generate_1d_report(fooof_outs):

    # Load data from fixture
    fm, fooof_graph = fooof_outs[0], fooof_outs[1]

    # Define output
    out_dir = TEST_DATA_PATH
    fname = 'ndspflow.html'

    # Embed plots
    subject = 'sub-001'
    generate_1d_report(fm, fooof_graph, subject, 0, 1, out_dir, fname)

    # Assert that the html file was generated
    assert os.path.isfile(os.path.join(out_dir, fname))

    # Cleanup
    os.remove(os.path.join(out_dir, fname))


def test_generate_header():

    subject = "sub-001"
    n_fooofs = 1
    n_bycycles = 0

    html_header = generate_header(subject, n_fooofs, n_bycycles)

    # Assert masthead template was used
    assert "masthead" in html_header

    # Assert subject template replacement was successful
    assert "{% SUBJECT_TEMPLATE %}" not in html_header
    assert subject in html_header
    assert str(n_fooofs) in html_header
    assert str(n_bycycles) in html_header


def test_generate_fooof_report(fooof_outs):

    # Load data from fixture
    fm, fooof_graph = fooof_outs[0], fooof_outs[1]

    html_report = generate_header('sub-001', 1, 0)
    html_report = generate_fooof_report(fm, fooof_graph, html_report)

    # Assert html template replacement was successful
    assert "{% fooof_settings %}" not in html_report
    assert "{% fooof_results %}" not in html_report
    assert "{% fooof_graph %}" not in html_report
