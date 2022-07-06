"""Configuration file for pytest for ndspflow."""

import os
import json
from shutil import rmtree

import numpy as np
import pytest

from neurodsp.sim import sim_combined
from neurodsp.spectral import compute_spectrum

from ndspflow.tests.settings import (N_SECONDS, FS, EXP, FREQ, F_RANGE)
from ndspflow.motif.fit import fit_bycycle

from fooof import FOOOF, FOOOFGroup

import numpy as np

from mne_bids import BIDSPath
from mne import create_info
from mne.io import RawArray
from mne_bids.write import write_raw_bids


@pytest.fixture(scope='module')
def sim_sig():

    # Simulate a 1d timeseries that contains an oscillation + 1/f
    sig = sim_combined(N_SECONDS, FS, {'sim_powerlaw': {'exponent': EXP},
                                       'sim_oscillation': {'freq': FREQ}})

    yield dict(n_seconds=N_SECONDS, fs=FS, exp=EXP, freq=FREQ, f_range=F_RANGE, sig=sig)


@pytest.fixture(scope='module')
def test_data(sim_sig):

    # Load data
    sig_1d = sim_sig['sig']
    fs = sim_sig['fs']
    f_range = sim_sig['f_range']

    # Duplicate sig to create 2d/3d arrays
    sig_2d = np.array([sig_1d] * 2)
    sig_3d = np.array([sig_2d] * 2)
    sig_4d = np.array([sig_3d] * 2)

    # FFT
    freqs, powers_1d = compute_spectrum(sig_1d, fs, f_range=f_range)

    # Create a (2, 100) array
    powers_2d = np.array([powers_1d for dim1 in range(2)])

    # Create a (2, 2, 100) array
    powers_3d =  np.array([[powers_1d for dim1 in range(2)] for dim2 in range(2)])

    # Create a (2, 2, 2, 100) array
    powers_4d = np.array([[[powers_1d for dim1 in range(2)] for dim2 in range(2)] \
        for dim3 in range(2)])

    yield {'sig_1d': sig_1d, 'sig_2d': sig_2d, 'sig_3d': sig_3d, 'sig_4d': sig_4d,
           'fs': fs, 'f_range': f_range, 'freqs': freqs,  'powers_1d': powers_1d,
           'powers_2d': powers_2d, 'powers_3d': powers_3d, 'powers_4d': powers_4d}


@pytest.fixture(scope='module')
def fooof_outs(test_data):

    # Load data
    powers_1d = test_data['powers_1d']
    powers_2d = test_data['powers_2d']

    freqs = test_data['freqs']
    f_range = test_data['f_range']

    # Fit
    fm = FOOOF(verbose=False)
    fm.fit(freqs, powers_1d, f_range)

    fg = FOOOFGroup(verbose=False)
    fg.fit(freqs, powers_2d, f_range)

    yield dict(fm=fm, fg=fg)


@pytest.fixture(scope='module')
def bycycle_outs(test_data):

    # Load data
    sig_1d = test_data['sig_1d']
    sig_2d = test_data['sig_2d']
    sig_3d = test_data['sig_3d']
    fs = test_data['fs']
    f_range = test_data['f_range']

    # Fit
    threshold_kwargs = dict(amp_fraction_threshold=0.5, amp_consistency_threshold=.5,
                            monotonicity_threshold=0.8, period_consistency_threshold=.5,
                            min_n_cycles=3)

    bm = fit_bycycle(sig_1d, fs, f_range, threshold_kwargs=threshold_kwargs)
    bg = fit_bycycle(sig_2d, fs, f_range, threshold_kwargs=threshold_kwargs)
    bgs = fit_bycycle(sig_3d, fs, f_range, threshold_kwargs=threshold_kwargs)

    yield dict(bm=bm, bg=bg, bgs=bgs)


@pytest.fixture(scope='module')
def motif_outs():

    comps_ref = {'sim_oscillation': {'freq' : 1, 'cycle': 'sine'}}
    motif_ref = sim_combined(1, 100, comps_ref)

    comps_target = {'sim_oscillation': {'freq' : 1, 'cycle': 'asine','rdsym': .6}}
    motif_target = sim_combined(1, 100, comps_target)

    yield dict(motif_ref=motif_ref, motif_target=motif_target)


@pytest.fixture(scope='module')
def bids_path():

    if os.path.isdir('_bids'):
        rmtree('_bids')

    os.mkdir('_bids')

    desc = {
        "Name": "Testing",
        "License": "CC0",
        "Authors": ["Some, Author"],
        "Acknowledgements": "Some, Name",
        "Funding": ["$"],
        "DatasetDOI": "xx.xxxx"
    }

    with open('_bids/dataset_description.json', 'w') as f:
        json.dump(desc, f)

    raw = RawArray(np.random.rand(2, 2000), create_info(['Oz', 'Pz'], 1000, 'ecog'), copy='both', verbose=False)
    _ = raw.set_channel_types({'Oz':'ecog', 'Pz':'ecog'})

    bpath = f'{os.getcwd()}/_bids'
    path = BIDSPath(root=bpath, subject='01', session='01', datatype='ieeg', task='test', extension='.vhdr')
    _ = write_raw_bids(raw, path, allow_preload=True, format='BrainVision', overwrite=True, verbose=False)

    yield bpath

    # Clean up bids tmp directory
    rmtree('_bids')


