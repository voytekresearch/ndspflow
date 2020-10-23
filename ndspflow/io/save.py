"""Functions to save results and plots."""

import os
import numpy as np

from fooof import FOOOF, FOOOFGroup
from ndspflow.core.utils import flatten_fms, flatten_bms
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
    fms, out_paths = flatten_fms(model, fooof_dir)

    # Save outputs
    for fm, out_path in zip(fms, out_paths):

        # Make the output directory
        clean_mkdir(out_path)

        # Save the model
        fm.save('results', file_path=out_path, append=False, save_results=True, save_settings=True)


def save_bycycle(df_features, df_samples, output_dir):
    """Make output directories and save bycycle dataframes.

    Parameters
    ----------
    model : FOOOF, FOOOFGroup, or list of FOOOFGroup objects.
        A FOOOF object that has been fit using :func:`ndspflow.core.fit.fit_fooof`.
    output_dir : str
        Path to write FOOOF results to.
    """

    # Make the bycycle output dir
    bycycle_dir = os.path.join(output_dir, 'bycycle')
    clean_mkdir(bycycle_dir)

    df_features, df_samples, bc_paths = flatten_bms(df_features, df_samples, output_dir)

    # Save outputs
    for df_feature, df_sample, bc_path in zip(df_features, df_samples, bc_paths):

        # Make the output directory
        clean_mkdir(bc_path)

        # Save the dataframes
        df_feature.to_csv(os.path.join(bc_path, 'results_features.csv'), index=False)
        df_sample.to_csv(os.path.join(bc_path, 'results_samples.csv'), index=False)
