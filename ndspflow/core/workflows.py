""""IO, FOOOF, and Bycycle workflow definitions."""

import numpy as np

from nipype import Workflow
import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu

from ndspflow.io.paths import check_dirs
from ndspflow.core.interfaces import FOOOFNode, BycycleNode


def create_workflow(input_dir, output_dir, run_nodes=['fooof', 'bycycle'],
                    fooof_params=None, bycycle_params=None, n_jobs=1):
    """Connects nodes into an overall nipype workflow.

    Parameters
    ----------
    input_dir : str
        Path to input directory.
    output_dir : str
        Path to write results to.
    run_nodes : list, optional, default: ['fooof', 'bycycle']
        Defines which nodes to run. Must contain fooof and/or bycycle.
    fooof_params : dict, optional, default: None
        Sets the inputs to the FOOOFNode.
    bycycle_params : dict, optional, default: None
        Sets the inputs to the BycycleNode.
    n_jobs : int
        The number of jobs to run in parrallel for 2d or 3d arrays.

    Returns
    -------
    wf : nipype.pipeline.engine.Workflow
        A nipype workflow containing fooof and/or bycycle sub-workflows.
    """
    # Setup entire workflow
    wf = pe.Workflow(name="wf_ndspflow")

    # Prepare read/write directories
    check_dirs(input_dir, output_dir)

    # Entry node
    io_node = pe.Node(niu.IdentityInterface(fields=['input_dir', 'output_dir', 'n_jobs']),
                      name='io_node')
    io_node.inputs.input_dir = input_dir
    io_node.inputs.output_dir = output_dir
    io_node.inputs.n_jobs = n_jobs

    # FOOOF node
    if 'fooof' in run_nodes:
        fooof_node = wf_fooof(fooof_params)
        wf.connect([(io_node, fooof_node, [('input_dir', 'input_dir'),
                                           ('output_dir', 'output_dir'),
                                           ('n_jobs', 'n_jobs')])])

    if 'bycycle' in run_nodes:
        bycycle_node = wf_bycycle(bycycle_params)
        wf.connect([(io_node, bycycle_node, [('input_dir', 'input_dir'),
                                             ('output_dir', 'output_dir'),
                                             ('n_jobs', 'n_jobs')])])

    return wf


def wf_fooof(fooof_params):
    """Create the fooof workflow.

    Parameters
    ----------
    fooof_params : dict
       Defines parameters in the init and fit methods of a FOOOF object.

    Returns
    -------
    fooof_node : nipype.pipeline.engine.nodes.Node
        A nipype node for running fooof.
    """

    # Check parameters
    if fooof_params is None:
        raise ValueError("Undefined required fooof parameters.")

    for param in ['freqs', 'power_spectrum']:
        if param not in fooof_params:
             raise ValueError("Undefined required fooof parameters.")

    # Create node
    fooof_node = pe.Node(FOOOFNode(), name='fooof_node')

    # Fit params
    fooof_node.inputs.freqs = fooof_params.pop('freqs')
    fooof_node.inputs.power_spectrum = fooof_params.pop('power_spectrum')
    fooof_node.inputs.f_range_fooof = fooof_params.pop('f_range_fooof', (-np.inf, np.inf))

    # Init params
    fooof_node.inputs.peak_width_limits = fooof_params.pop('peak_width_limits', (0.5, 12.0))
    fooof_node.inputs.max_n_peaks = fooof_params.pop('max_n_peaks', 100)
    fooof_node.inputs.min_peak_height = fooof_params.pop('min_peak_height', 0.0)
    fooof_node.inputs.peak_threshold = fooof_params.pop('peak_threshold', 2.0)
    fooof_node.inputs.aperiodic_mode = fooof_params.pop('aperiodic_mode', 'fixed')

    return fooof_node


def wf_bycycle(bycycle_params):
    """Create the bycycle workflow.

    Parameters
    ----------
    bycycle_params : dict
        Defines parameters for use in bycycle's ``compute_features``.

    Returns
    -------
    bycycle_node : nipype.pipeline.engine.nodes.Node
        A nipype node for running bycycle.
    """

    # Check parameters
    if bycycle_params is None:
        raise ValueError("Undefined required bycycle parameters.")

    for param in ['sig', 'fs', 'f_range_bycycle']:
        if param not in bycycle_params:
             raise ValueError("Undefined required bycycle parameters.")

    # Create node
    bycycle_node = pe.Node(BycycleNode(), name='bycycle_node')

    # Required arguments
    bycycle_node.inputs.sig = bycycle_params.pop("sig")
    bycycle_node.inputs.fs = bycycle_params.pop("fs")
    bycycle_node.inputs.f_range_bycycle = bycycle_params.pop("f_range_bycycle")

    # Optional arguments
    bycycle_node.inputs.center_extrema = bycycle_params.pop("center_extrema", "peak")
    bycycle_node.inputs.burst_method = bycycle_params.pop("burst_method", "cycles")
    bycycle_node.inputs.amp_fraction_threshold = bycycle_params.pop("amp_fraction_threshold", 0)
    bycycle_node.inputs.amp_consistency_threshold = \
        bycycle_params.pop("amp_consistency_threshold", 0.5)
    bycycle_node.inputs.period_consistency_threshold = \
        bycycle_params.pop("period_consistency_threshold", 0.5)
    bycycle_node.inputs.monotonicity_threshold = bycycle_params.pop("monotonicity_threshold", 0.8)
    bycycle_node.inputs.min_n_cycles = bycycle_params.pop("min_n_cycles", 3)
    bycycle_node.inputs.burst_fraction_threshold = \
        bycycle_params.pop("burst_fraction_threshold", 1.0)
    bycycle_node.inputs.axis = bycycle_params.pop("axis", 0)

    return bycycle_node
