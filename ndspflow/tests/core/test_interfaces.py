"""Test FOOOOF and Bycycle nipype interfaces."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import nipype.pipeline.engine as pe

from ndspflow.tests.settings import TEST_DATA_PATH
from ndspflow.core.interfaces import FOOOFNode


def test_FOOOF():

    fooof = FOOOFNode()

    # Check required params
    assert hasattr(fooof.inputs, 'power_spectrum')
    assert hasattr(fooof.inputs, 'input_dir')
    assert hasattr(fooof.inputs, 'output_dir')
    assert hasattr(fooof.inputs, 'freqs')

    # Check default params
    assert fooof.inputs.aperiodic_mode == 'fixed'
    assert fooof.inputs.f_range_fooof == (-np.inf, np.inf)
    assert fooof.inputs.max_n_peaks == 100
    assert fooof.inputs.min_peak_height == 0.0
    assert fooof.inputs.peak_threshold == 2.0
    assert fooof.inputs.peak_width_limits == (0.5, 12.0)

    # Check results, ensure empty on initialization
    assert type(fooof._results) is dict
    assert len(fooof._results) == 0

    # Check running the interface on a node
    fooof_node = pe.Node(fooof, name='fooof_node')
    fooof_node.inputs.input_dir = TEST_DATA_PATH
    fooof_node.inputs.freqs = 'freqs.npy'
    fooof_node.inputs.power_spectrum = 'spectrum.npy'

    # Setup a temporary director to write to
    test_dir = TemporaryDirectory()
    fooof_node.inputs.output_dir = test_dir.name

    # Run the node and assert output exists
    fooof_node.run()
    f_out = os.listdir(os.path.join(test_dir.name, 'fooof'))
    assert 'results.json' in f_out
    assert 'report.html' in f_out
    test_dir.cleanup()
