#!/usr/bin/env python3
"""Commmand-line interface for ndspflow."""

import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from ndspflow.core.workflows import create_workflow

import warnings
warnings.filterwarnings("ignore")


def get_parser():
    """Parse command-line arguments."""

    import argparse, textwrap

    desc = 'A Nipype workflow for FOOOOF and Bycycle.'
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.RawTextHelpFormatter)
    # I/O
    parser.add_argument(
        'input_dir',
        type=str,
        default=None,
        metavar="/path/to/input",
        help='Input directory containing timeseries and/or spectra .npy files to read (default: %(default)s).'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        default=None,
        metavar="/path/to/output",
        help='Output directory to write results and BIDS derivatives to write (default: %(default)s).'
    )

    # FOOOF fit params
    parser.add_argument(
        '-power_spectrum',
        type=str,
        default=None,
        metavar="powers.npy",
        help="Filename of power values, located inside of 'input_dir'\n"
             "Required if 'fooof' in 'run_nodes argument' (default: %(default)s)."

    )
    parser.add_argument(
        '-freqs',
        type=str,
        default=None,
        metavar="freqs.npy",
        help="Filename of frequency values for the power spectrum(a), located inside of 'input_dir'.\n"
             "Required if 'fooof' in 'run_nodes argument' (default: %(default)s)."
    )
    parser.add_argument(
        '-freq_range',
        type=float,
        nargs=2,
        default=(-np.inf, np.inf),
        metavar=("lower_freq", "upper_freq"),
        help="Frequency range of the power spectrum, as: lower_freq, upper_freq.\n"
             "Recommended if 'fooof' in 'run_nodes argument' (default: %(default)s)."
    )
    parser.add_argument(
        '-sig',
        type=str,
        default=None,
        metavar="signal.npy",
        help="Filename of neural signal or timeseries, located inside of 'input_dir'.\n"
             "Required if 'bycycle' in 'run_nodes argument' (default: %(default)s).\n "
    )

    # FOOOF init params
    parser.add_argument(
        '-peak_width_limits',
        type=float,
        nargs=2,
        default=(0.5, 12.0),
        metavar=("lower_limit", "upper_limit"),
        help="Limits on possible peak width, in Hz, as: lower_limit upper_limit.\n"
             "Recommended if 'fooof' in 'run_nodes argument' (default: %(default)s)."
    )
    parser.add_argument(
        '-max_n_peaks',
        type=int,
        default=100,
        metavar="int",
        help="Maximum number of peaks to fit.\n"
             "Recommended if 'fooof' in 'run_nodes argument' (default: %(default)s)."
    )
    parser.add_argument(
        '-min_peak_height',
        type=float,
        default=0.,\
        metavar="float",
        help="Absolute threshold for detecting peaks, in units of the input data.\n"
             "Recommended if 'fooof' in 'run_nodes argument' (default: %(default)s)."
    )
    parser.add_argument(
        '-peak_threshold',
        type=float,
        default=2.0,
        metavar="float",
        help="Relative threshold for detecting peaks, in units of standard deviation of the input data.\n"
             "Recommended if 'fooof' in 'run_nodes argument' (default: %(default)s)."

    )
    parser.add_argument(
        '-aperiodic_mode',
        type=str,
        default='fixed',
        choices=['fixed', 'knee'],
        help="Which approach to take for fitting the aperiodic component.\n"
             "Recommended if 'fooof' in 'run_nodes argument' (default: %(default)s)."
    )
    parser.add_argument(
        '-n_jobs',
        type=int,
        metavar="int",
        default=1,
        help="The maximum number of jobs to run in parallel at one time.\n"
             "Only utilized for 2d and 3d arrays (default: %(default)s)."
    )

    # Workflow selector
    parser.add_argument(
        '-run_nodes',
        default=['fooof', 'bycycle'],
        choices=['fooof', 'bycycle'],
        required=False,
        help="List of nodes to run: fooof and/or bycyle (default: fooof bycycle)."
    )

    return parser


def main():

    # Get arguments
    args = get_parser().parse_args()
    args = vars(args)

    input_dir = args['input_dir']
    output_dir = args['output_dir']

    run_nodes = args['run_nodes']

    if 'fooof' in run_nodes:
        fooof_params = {}
        fooof_params['freqs'] = args['freqs']
        fooof_params['power_spectrum'] = args['power_spectrum']
        fooof_params['freq_range'] = args['freq_range']
        fooof_params['peak_width_limits'] = args['peak_width_limits']
        fooof_params['max_n_peaks'] = args['max_n_peaks']
        fooof_params['min_peak_height'] = args['min_peak_height']
        fooof_params['peak_threshold'] = args['peak_threshold']
        fooof_params['aperiodic_mode'] = args['aperiodic_mode']
    else:
        fooof_params = None

    if 'bycycle' in run_nodes:
        bycycle_params = {}
    else:
        bycycle_params=None

    n_jobs = args['n_jobs']

    wf = create_workflow(input_dir, output_dir, run_nodes=run_nodes, fooof_params=fooof_params,
                         bycycle_params=bycycle_params, n_jobs=n_jobs)


    wf.run()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
