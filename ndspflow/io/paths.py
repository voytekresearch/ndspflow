"""Input/output utility functions.""""

import os


def setup_dirs(input_dir, output_dir):
    """Create input/out directories.

    Parameters
    ----------
    input_dir : str
        Path to input BIDS directory.
    output_dir : str
        Path to write results or BIDS deriavates to.
    """

    if not os.path.isdir(input_dir):
        raise ValueError("Input BIDS directory does not exist.")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
