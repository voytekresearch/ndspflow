"""Configuration file for pytest for ndspflow."""

import os
import numpy as np
import pytest

from ndspflow.tests.settings import TEST_DATA_PATH
from fooof import FOOOF
from ndspflow.core.fit import fit_fooof
from ndspflow.plts.fooof import plot_fm, plot_fg, plot_fgs


@pytest.fixture(scope='module')
def test_data():

    # Load data
    powers_1d = np.load(os.path.join(TEST_DATA_PATH, 'spectrum.npy'))
    freqs = np.load(os.path.join(TEST_DATA_PATH, 'freqs.npy'))

    # Create a (2, 100) array
    powers_2d = np.array([powers_1d for dim1 in range(2)])

    # Create a (2, 2, 100) array
    powers_3d =  np.array([[powers_1d for dim1 in range(2)] for dim2 in range(2)])

    # Create a (2, 2, 2, 100) array
    powers_4d = np.array([[[powers_1d for dim1 in range(2)] for dim2 in range(2)] \
        for dim3 in range(2)])

    yield {'freqs': freqs, 'powers_1d': powers_1d, 'powers_2d': powers_2d,
           'powers_3d': powers_3d, 'powers_4d': powers_4d}


@pytest.fixture(scope='module')
def fm_outs(input_dir=TEST_DATA_PATH):

    # Load data
    freqs = np.load(os.path.join(input_dir, 'freqs.npy'))
    spectrum = np.load(os.path.join(input_dir, 'spectrum.npy'))
    freq_range = (1, 40)

    # Fit
    fm = FOOOF(peak_width_limits=(0.5, 12.0), max_n_peaks=np.inf, min_peak_height=0.0,
               peak_threshold=2.0, aperiodic_mode='fixed', verbose=False)

    fm.fit(freqs, spectrum, freq_range)

    # Plot
    fooof_graph = plot_fm(fm)

    yield {'fm': fm, 'fooof_graph': fooof_graph}


@pytest.fixture(scope='module')
def fg_outs(input_dir=TEST_DATA_PATH):

    # Load data
    powers_1d = np.load(os.path.join(TEST_DATA_PATH, 'spectrum.npy'))
    freqs = np.load(os.path.join(TEST_DATA_PATH, 'freqs.npy'))

    # Create a (2, 100) array
    powers_2d = np.array([powers_1d for dim1 in range(2)])

    # Fit
    fg = fit_fooof(freqs, powers_2d, (1, 40), {}, 1)

    # Plot
    fooof_graph = plot_fg(fg, ['' for i in range(len(fg))])

    yield {'fg': fg, 'fooof_graph': fooof_graph}


@pytest.fixture(scope='module')
def fgs_outs(input_dir=TEST_DATA_PATH):

    # Load data
    powers_1d = np.load(os.path.join(TEST_DATA_PATH, 'spectrum.npy'))
    freqs = np.load(os.path.join(TEST_DATA_PATH, 'freqs.npy'))

    # Create a (2, 100) array
    powers_2d = np.array([powers_1d for dim1 in range(2)])

    # Create a (2, 2, 100) array
    powers_3d =  np.array([[powers_1d for dim1 in range(2)] for dim2 in range(2)])

    # Fit
    fgs = fit_fooof(freqs, powers_3d, (1, 40), {}, 1)

    # Plot
    fooof_graph = plot_fgs(fgs, ['' for i in range(int(len(fgs)*len(fgs[0])))])

    yield {'fgs': fgs, 'fooof_graph': fooof_graph}


