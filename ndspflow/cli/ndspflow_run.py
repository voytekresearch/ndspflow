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

    # FOOOF required params
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
        '-f_range_fooof',
        type=float,
        nargs=2,
        default=(-np.inf, np.inf),
        metavar=("lower_freq", "upper_freq"),
        help="Frequency range of the power spectrum, as: lower_freq, upper_freq.\n"
             "Recommended if 'fooof' in 'run_nodes argument' (default: %(default)s)."
    )

    # Bycycle required params
    parser.add_argument(
        '-sig',
        type=str,
        default=None,
        metavar="signal.npy",
        help="Filename of neural signal or timeseries, located inside of 'input_dir'.\n"
             "Required if 'bycycle' in 'run_nodes argument' (default: %(default)s)."
    )
    parser.add_argument(
        '-fs',
        type=int,
        default=None,
        metavar="int",
        help="Sampling rate, in Hz.\n"
             "Required if 'bycycle' in 'run_nodes argument'."
    )
    parser.add_argument(
        '-f_range_bycycle',
        type=float,
        nargs=2,
        default=None,
        metavar=("lower_freq", "upper_freq"),
        help="Frequency range for narrowband signal of interest (Hz).\n"
             "Required if 'bycycle' in 'run_nodes argument'.\n "
    )

    # FOOOF optional params
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
             "Recommended if 'fooof' in 'run_nodes argument' (default: %(default)s).\n "
    )

    # Bycycle optional arguments
    parser.add_argument(
        '-center_extrema',
        type=str,
        default='peak',
        choices=['peak', 'trough'],
        help="Determines if cycles or peak or trough centered.\n"
             "Recommended if 'bycycle' in 'run_nodes argument' (default: %(default)s)."
    )

    # Burst method and thresholds
    parser.add_argument(
        '-burst_method',
        type=str,
        default='cycles',
        choices=['cycles', 'amp'],
        help="Method for burst detection.\n"
             "Recommended if 'bycycle' in 'run_nodes argument' (default: %(default)s)."
    )
    parser.add_argument(
        '-amp_fraction_threshold',
        type=float,
        metavar="float",
        default=0,
        help="Amplitude fraction threshold for detecting bursts.\n"
             "Recommended if 'burst_method' is 'cycles' (default: %(default)s)."
    )
    parser.add_argument(
        '-amp_consistency_threshold',
        type=float,
        metavar="float",
        default=0.5,
        help="Amplitude consistency threshold for detecting bursts.\n"
             "Recommended if 'burst_method' is 'cycles' (default: %(default)s)."
    )
    parser.add_argument(
        '-period_consistency_threshold',
        type=float,
        metavar="float",
        default=0.5,
        help="Period consistency threshold for detecting bursts.\n"
             "Recommended if 'burst_method' is 'cycles' (default: %(default)s)."
    )
    parser.add_argument(
        '-monotonicity_threshold',
        type=float,
        metavar="float",
        default=0.8,
        help="Monotonicicity threshold for detecting bursts.\n"
             "Recommended if 'burst_method' is 'cycles' (default: %(default)s)."
    )
    parser.add_argument(
        '-min_n_cycles',
        type=int,
        metavar="int",
        default=3,
        help="Minium number of cycles for detecting bursts\n"
             "Recommended for either 'burst_method' (default: %(default)s)."
    )
    parser.add_argument(
        '-burst_fraction_threshold',
        type=float,
        metavar="float",
        default=1,
        help="Minimum fraction of a cycle identified as a burst.\n"
             "Recommended if 'burst_method' is 'amp' (default: %(default)s)."
    )
    parser.add_argument(
        '-axis',
        type=str,
        metavar="{0, 1, (0, 1), None}",
        default='0',
        help="The axis to compute features across for 2D and 3D signal arrays.\n"
             "Ignored if signal is 1D. 1 and (0, 1) only availble for 3D signals\n"
             "(default: %(default)s).\n "
    )

    # Parallel computation
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

        fooof_params = dict(
            freqs=args['freqs'], power_spectrum=args['power_spectrum'],
            f_range_fooof=args['f_range_fooof'], peak_width_limits=args['peak_width_limits'],
            max_n_peaks=args['max_n_peaks'], min_peak_height=args['min_peak_height'],
            peak_threshold=args['peak_threshold'], aperiodic_mode=args['aperiodic_mode']
        )

    else:
        fooof_params = None

    if 'bycycle' in run_nodes:

        bycycle_params = dict(
            sig=args['sig'], fs=args['fs'], f_range_bycycle=args['f_range_bycycle'],
            center_extrema=args['center_extrema'], burst_method=args['burst_method'],
            amp_fraction_threshold=args['amp_fraction_threshold'],
            amp_consistency_threshold=args['amp_consistency_threshold'],
            period_consistency_threshold=args['period_consistency_threshold'],
            monotonicity_threshold=args['monotonicity_threshold'],
            burst_fraction_threshold=args['burst_fraction_threshold'],
            min_n_cycles=args['min_n_cycles'], axis=args['axis']
        )

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
