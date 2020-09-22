#!/usr/bin/env python3
"""Commmand-line interface for ndspflow."""

import numpy as np
from workflows import create_workflow
import warnings
warnings.filterwarnings("ignore")

def get_parser():
    """Parse command-line arguments"""

    import argparse

    desc = 'A Workflow for FOOOOF and Bycycle.'
    parser = argparse.ArgumentParser(description=desc)

    # I/O
    parser.add_argument(
        'input_dir',
        type=str,
        default=None,
        help='Input directory containing timeseries and/or spectra .npy files to '\
                'read.\n'
    )
    parser.add_argument(
        'output_dir',
        default=None,
        type=str,
        help='Output directory to write results and BIDS derivatives to write.\n'
    )

     # FOOOF fit params
    parser.add_argument(
        '-freqs',
        type=str,
        help="FOOOF: Frequency values for the power spectrum."
)
    parser.add_argument(
        '-power_spectrum',
        type=str,
        default=None,
        help="FOOOF: Power values, stored internally in log10 scale."
    )
    parser.add_argument(
        '-freq_range',
        type=float,
        nargs=2,
        default=None,
        help="FOOOF: Frequency range of the power spectrum, as: lowest_freq, highest_freq."
    )

    # FOOOF init params
    parser.add_argument(
        '-peak_width_limits',
        type=float,
        nargs=2,
        default=[0.5, 12.0],
        help="FOOOF: Limits on possible peak width, in Hz, as: lower_bound upper_bound."
    )
    parser.add_argument(
        '-max_n_peaks',
        type=int,
        default=np.inf,
        help="FOOOF: Maximum number of peaks to fit."
    )
    parser.add_argument(
        '-min_peak_height',
        type=float,
        default=0.,
        help="FOOOF: Absolute threshold for detecting peaks, in units of the input data."
    )
    parser.add_argument(
        '-peak_threshold',
        type=float,
        default=2.0,
        help="FOOOF: Relative threshold for detecting peaks, in units of standard deviation of the input data."
    )
    parser.add_argument(
        '-periodic_mode',
        type=str,
        default='fixed',
        help="FOOOF: Which approach to take for fitting the aperiodic component."
    )

    parser.add_argument(
        '-run_nodes',
        default=['fooof', 'bycycle'],
        nargs='+',
        required=False,
        help=f"List of nodes to run: fooof and/or bycyle"
    )

    return parser


def main():

    # Get arguments
    args = get_parser().parse_args()
    args = vars(args)

    input_dir = args['input_dir'][0]
    output_dir = args['output_dir'][0]

    fooof_params = {}
    fooof_params['freqs'] = args['freqs'][0]
    fooof_params['power_spectrum'] = args['power_spectrum'][0]
    fooof_params['freq_range'] = args['freq_range'][0]

    run_nodes = args['run_nodes']

    #### Add other params here. ###

    wf = create_workflow(input_dir, output_dir, run_nodes=run_nodes)
    wf = create_workflow(input_dir, output_dir, run_nodes=run_nodes, fooof_params=fooof_params)
    wf.run()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
