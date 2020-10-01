"""Configuration file for pytest for ndspflow."""

import os
import numpy as np
import pytest

from ndspflow.tests.settings import TEST_DATA_PATH


@pytest.fixture(scope='module')
def test_data():

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
