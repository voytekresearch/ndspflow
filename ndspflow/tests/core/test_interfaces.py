"""Test FOOOOF and Bycycle nipype interfaces."""

import os
from tempfile import TemporaryDirectory
from pytest import mark, param

import numpy as np
import nipype.pipeline.engine as pe

from ndspflow.tests.settings import TEST_DATA_PATH
from ndspflow.core.interfaces import FOOOFNode, BycycleNode, ReportNode


def test_FOOOF():

    fooof = FOOOFNode()

    # Check required params
    assert hasattr(fooof.inputs, 'power_spectrum')
    assert hasattr(fooof.inputs, 'input_dir')
    assert hasattr(fooof.inputs, 'output_dir')
    assert hasattr(fooof.inputs, 'freqs')

    # Check optional params
    assert fooof.inputs.aperiodic_mode == 'fixed'
    assert fooof.inputs.f_range_fooof == (-np.inf, np.inf)
    assert fooof.inputs.max_n_peaks == 100
    assert fooof.inputs.min_peak_height == 0.0
    assert fooof.inputs.peak_threshold == 2.0
    assert fooof.inputs.peak_width_limits == (0.5, 12.0)

    # Check results, ensure empty on initialization
    assert isinstance(fooof._results, dict)
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
    test_dir.cleanup()



@mark.parametrize("axis", ['None', param('2', marks=mark.xfail), param('X', marks=mark.xfail)])
@mark.parametrize("burst_method", ['cycles', 'amp'])
def test_Bycycle(axis, burst_method):

    bycycle = BycycleNode()

    # Check required params
    assert hasattr(bycycle.inputs, 'sig')
    assert hasattr(bycycle.inputs, 'fs')
    assert hasattr(bycycle.inputs, 'f_range_bycycle')
    assert hasattr(bycycle.inputs, 'input_dir')
    assert hasattr(bycycle.inputs, 'output_dir')

    # Check optional params
    assert bycycle.inputs.center_extrema == 'peak'
    assert bycycle.inputs.burst_method == 'cycles'
    assert bycycle.inputs.amp_fraction_threshold == 0.0
    assert bycycle.inputs.amp_consistency_threshold == 0.5
    assert bycycle.inputs.period_consistency_threshold == 0.5
    assert bycycle.inputs.monotonicity_threshold == 0.8
    assert bycycle.inputs.min_n_cycles == 3
    assert bycycle.inputs.burst_fraction_threshold == 1.0
    assert bycycle.inputs.axis == 'None'
    assert bycycle.inputs.n_jobs == 1

    # Check results, ensure empty on initialization
    assert isinstance(bycycle._results, dict)
    assert len(bycycle._results) == 0

    # Check running the interface on a node
    bycycle_node = pe.Node(bycycle, name='bycycle_node')
    bycycle_node.inputs.input_dir = TEST_DATA_PATH
    bycycle_node.inputs.sig = 'sig.npy'
    bycycle_node.inputs.fs = 500
    bycycle_node.inputs.f_range_bycycle = (1, 40)
    bycycle_node.inputs.axis = axis
    bycycle_node.inputs.burst_method = burst_method

    # Setup a temporary director to write to
    test_dir = TemporaryDirectory()
    bycycle_node.inputs.output_dir = test_dir.name

    # Run the node and assert output exists
    bycycle_node.run()
    f_out = os.listdir(os.path.join(test_dir.name, 'bycycle'))
    assert 'results.csv' in f_out
    test_dir.cleanup()


def test_Report():

    report_node = ReportNode()

    report_node.inputs.output_dir = '/tmp'
    report_node.inputs.fms = None
    report_node.inputs.df_features = None
    report_node.inputs._fit_args = None

    report_node.run()
