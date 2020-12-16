"""Utility functions for organizing fooof and bycycle outputs."""

import os
import numpy as np
import pandas as pd
from fooof import FOOOF, FOOOFGroup, fit_fooof_3d


def flatten_fms(model, output_dir):
    """Flatten various oranizations of fooof models into a 1d list.

    Parameters
    ----------
    model : FOOOF, FOOOFGroup, or list of FOOOFGroup objects.
        A FOOOF object that has been fit using :func:`ndspflow.core.fit.fit_fooof`.
    output_dir : str
        Path to write FOOOF results to.

    Returns
    -------
    fms : list of fooof FOOOF
        A flattened list of FOOOF objects.
    fm_paths : list of str
        Sub-directories to write fooof reports to.
    """

    # Flatten the models and output dirs into a 1d list
    fms = []
    fm_paths = []

    if type(model) is FOOOF:

        # For 1d arrays results
        fm_paths.append(output_dir)
        fms.append(model)

    elif type(model) is FOOOFGroup:

        # For 2d arrays results
        label_template = "spectrum_dim1-{dim_a}"
        for fm_idx in range(len(model)):

            label = label_template.format(dim_a=str(fm_idx).zfill(4))
            fm_paths.append(os.path.join(output_dir, label))
            fms.append(model.get_fooof(fm_idx))

    elif type(model) is list:

        # For 3d arrays results
        label_template = "spectrum_dim1-{dim_a}_dim2-{dim_b}"

        for fg_idx in range(len(model)):

            for fm_idx in range(len(model[0].get_results())):

                label = label_template.format(dim_a=str(fg_idx).zfill(4),
                                              dim_b=str(fm_idx).zfill(4))
                fm_paths.append(os.path.join(output_dir, label))
                fms.append(model[fg_idx].get_fooof(fm_idx))

    return fms, fm_paths


def flatten_bms(df_features, output_dir, sigs=None):
    """Flatten various oranizations of bycycle dataframes into a 1d list.

    Parameters
    ----------
    df_features : pandas.DataFrame or list of pandas.DataFrame
        Dataframes containing shape and burst features for each cycle.
    output_dir : str
        Path to write bycycle results to.
    sigs : 1d, 2d, or 3d array, optional, default: None
        Voltage time series.

    Returns
    -------
    df_features : 1d list of pandas.DataFrame
        Dataframes containing shape and burst features for each cycle.
    bc_paths : list of str
        Sub-directories to write bycycle dataframes to.
    sigs_2d : 2d list
        A 2d arangement of sigs. None if sigs is None.
    """

    bc_paths = []

    if isinstance(df_features, pd.DataFrame):

        # For 1d array results
        bc_paths.append(output_dir)

        # Make dataframe an iterable list
        df_features = [df_features]

        sigs_2d = np.array([sigs]) if isinstance(sigs, np.ndarray) else None

    elif isinstance(df_features, list) and isinstance(df_features[0], pd.DataFrame):

        # For 2d array results
        label_template = "signal_dim1-{dim_a}"
        for fm_idx in range(len(df_features)):

            label = label_template.format(dim_a=str(fm_idx).zfill(4))
            bc_paths.append(os.path.join(output_dir, label))

        sigs_2d = sigs.copy() if isinstance(sigs, np.ndarray) else None

    elif isinstance(df_features, list) and isinstance(df_features[0][0], pd.DataFrame):

        # For 3d arrays results
        label_template = "signal_dim1-{dim_a}_dim2-{dim_b}"

        for bg_idx in range(len(df_features)):

            for bc_idx in range(len(df_features[bg_idx])):

                label = label_template.format(dim_a=str(bg_idx).zfill(4),
                                              dim_b=str(bc_idx).zfill(4))
                bc_paths.append(os.path.join(output_dir, label))

        df_features = [df for dfs in df_features for df in dfs]

        sigs_2d = np.reshape(sigs, (np.shape(sigs)[0] * np.shape(sigs)[1], np.shape(sigs)[2])) if \
            isinstance(sigs, np.ndarray) else None

    return df_features, bc_paths, sigs_2d
