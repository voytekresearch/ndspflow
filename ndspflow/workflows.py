""""IO, FOOOF, and Bycycle workflow definitions."""

from nipype import Workflow
import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu

from io.paths import setup_dirs
from interfaces import FOOOF



def create_workflow(input_dir, output_dir, run_nodes=['fooof', 'bycycle'], fooof_params=None):
    """Connects nodes into an overall nipype workflow.

    Parameters
    ----------
    input_dir : str
        Path to input BIDS directory.
    output_dir : str
        Path to write results or BIDS deriavates to.
    run_nodes : list, optional, default: ['fooof', 'bycycle']
        Names of nodes to add to workflow.
    fooof_params : dict, optional, default: None
        Parameters to pass into the init and fit methods of a fooof object.

    Returns
    -------
    wf : nipype.pipeline.engine.Workflow
        A nipype workflow containing fooof and/or bycycle subflows.
    """
    # Setup entire workflow
    wf = pe.Workflow(name="wf_ndspflow")

    # Prepare read/write directories
    setup_dirs(input_dir, outpt_dir)

    io_node = pe.Node(niu.IdentityInterface(fields=['input_dir', 'output_dir']),
                      name='input_node')
    io_node.inputs.input_dir = input_dir
    io_node.inputs.output_dir = output_dir

    wf.add_nodes(io_node)

    if 'fooof' in run_nodes:

        fooof_node = wf_fooof(fooof_params)

        wf.add_nodes(fooof_node)
        wf.connect([(input_node, fooof_node, [('input_dir', 'input_dir'),
                                              ('output_dir', 'output_dir')])])

    return wf


def wf_fooof(fooof_params):
    """Create the FOOOOF workflow.

    Parameters
    ----------
    fooof_params : list
        List of parameters definied in the init and fit methods of a FOOOF object.

    Returns
    -------
    fooof_node : nipype.pipeline.engine.nodes.Node
        A nipype node for running FOOOOF.
    """

    fooof_node = pe.Node(niu.IdentityInterface(fields=fooof_params), name='fooof_node')

    # Fit params
    fooof_node.freqs = foooof_params.pop('freqs', None)
    fooof_node.power_spectrum = foooof_params.pop('power_spectrum', None)
    fooof_node.freq_range = foooof_params.pop('freq_range', (-np.inf, np.inf))

    # Init params
    fooof_node.peak_width_limits = foooof_params.pop('peak_width_limits', (0.5, 12.0))
    fooof_node.max_n_peaks = foooof_params.pop('max_n_peaks', np.inf)
    fooof_node.peak_width_limits = foooof_params.pop('min_peak_height', 0.0)
    fooof_node.peak_threshold = foooof_params.pop('peak_threshold', 2.0)
    fooof_node. periodic_mode = foooof_params.pop(' periodic_mode', 'fixed')

    return fooof_node
