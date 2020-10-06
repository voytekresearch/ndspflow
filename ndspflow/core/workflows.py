""""IO, FOOOF, and Bycycle workflow definitions."""

import numpy as np

from nipype import Workflow
import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu

from ndspflow.io.paths import check_dirs
from ndspflow.core.interfaces import FOOOF


def create_workflow(input_dir, output_dir, run_nodes=['fooof', 'bycycle'],
                    fooof_params=None, bycyle_params=None, n_jobs=1):
    """Connects nodes into an overall nipype workflow.

    Parameters
    ----------
    input_dir : str
        Path to input directory.
    output_dir : str
        Path to write results to.
    run_nodes : list, optional, default: ['fooof', 'bycycle']
        Names of nodes to add to workflow.
    fooof_params : dict, optional, default: None
        Parameters to pass into the init and fit methods of a fooof object.
    bycycle_params : dict, optional, default: None
        Parameters to passing into a bycycle fit.
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

    return wf


def wf_fooof(fooof_params):
    """Create the fooof workflow.

    Parameters
    ----------
    fooof_params : list
        List of parameters defined in the init and fit methods of a FOOOF object.

    Returns
    -------
    fooof_node : nipype.pipeline.engine.nodes.Node
        A nipype node for running fooof.
    """

    fooof_params = {} if type(fooof_params) is None else fooof_params
    fooof_node = pe.Node(FOOOF(), name='fooof_node')

    # Fit params
    fooof_node.inputs.freqs = fooof_params.pop('freqs', None)
    fooof_node.inputs.power_spectrum = fooof_params.pop('power_spectrum', None)
    fooof_node.inputs.freq_range = fooof_params.pop('freq_range', (-np.inf, np.inf))

    # Init params
    fooof_node.inputs.peak_width_limits = fooof_params.pop('peak_width_limits', (0.5, 12.0))
    fooof_node.inputs.max_n_peaks = fooof_params.pop('max_n_peaks', 100)
    fooof_node.inputs.min_peak_height = fooof_params.pop('min_peak_height', 0.0)
    fooof_node.inputs.peak_threshold = fooof_params.pop('peak_threshold', 2.0)
    fooof_node.inputs.aperiodic_mode = fooof_params.pop('aperiodic_mode', 'fixed')

    return fooof_node
