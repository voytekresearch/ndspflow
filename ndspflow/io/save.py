"""Functions to save results and plots."""

import os
import numpy as np

from fooof import FOOOF, FOOOFGroup
from ndspflow.core.fit import flatten_fms
from ndspflow.io.paths import clean_mkdir


def save_fooof(model, output_dir):
    """Make output directories and save FOOOF fits.

    Parameters
    ----------
    model : FOOOF, FOOOFGroup, or list of FOOOFGroup objects.
        A FOOOF object that has been fit using :func:`ndspflow.core.fit.fit_fooof`.
    output_dir : str
        Path to write FOOOF results to.
    """

    # Make the fooof output dir
    fooof_dir = os.path.join(output_dir, 'fooof')
    clean_mkdir(fooof_dir)

    # Flatten model(s) and create output paths and sub-dir labels
    fms, out_paths, labels = flatten_fms(model, fooof_dir)

    # Save outputs
    for fm, out_path, label in zip(fms, out_paths, labels):

        # Make the output directory
        clean_mkdir(out_path)

        # Save the model
        fm.save('results', file_path=out_path, append=False, save_results=True, save_settings=True)
