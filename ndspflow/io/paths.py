"""Input/output utility functions."""

import os


def check_dirs(input_dir, output_dir):
    """Check input/out directories.

    Parameters
    ----------
    input_dir : str
        Path to input BIDS directory.
    output_dir : str
        Path to write results or BIDS deriavates to.
    """

    if not os.path.isdir(input_dir):
        raise ValueError("Input directory does not exist.")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
