"""Input/output utility functions."""

import os
from shutil import rmtree


def check_dirs(input_dir, output_dir):
    """Check input/out directories.

    Parameters
    ----------
    input_dir : str
        Absolute path to input BIDS directory.
    output_dir : str
        Absolute path to write results or BIDS deriavates to.
    """

    if not os.path.isdir(input_dir):
        raise ValueError("Input directory does not exist.")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


def clean_mkdir(dir_path):
    """Cleanly make new directories by deleting pre-existing.

    Parameters
    ----------
    dir_path : str
       Absoulte directory location path to clean.
    """

    if os.path.isdir(dir_path):
        rmtree(os.path.join(dir_path))

    os.makedirs(dir_path)
