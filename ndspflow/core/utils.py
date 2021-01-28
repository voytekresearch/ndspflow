"""Utility functions for organizing fooof and bycycle outputs."""


import os

import numpy as np
import pandas as pd

from fooof import FOOOF, FOOOFGroup


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

    # Type is required over isinstance - we don't want class inheritance checks
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

        for fg_idx, fg in enumerate(model):

            for fm_idx, _ in enumerate(model[0].get_results()):

                label = label_template.format(dim_a=str(fg_idx).zfill(4),
                                              dim_b=str(fm_idx).zfill(4))
                fm_paths.append(os.path.join(output_dir, label))
                fms.append(fg.get_fooof(fm_idx))

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


def limit_df(df_features, fs, f_range, only_bursts=True):
    """Limit a bycycle dataframe to a frequency range.

    Parameters
    ----------
    df_features : pandas.DataFrame
        A dataframe containing cycle features.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        The frequency range of interest.
    only_bursts : bool, optional, default: True
        Limits the dataframe to bursting cycles when True.

    Returns
    -------
    df_filt : pandas.DataFrame
        A dataframe containing cycle features within ``f_range``.
    """

    # Filter by bursting cycles if requested
    if only_bursts:
        df_filt = df_features.iloc[np.where(df_features['is_burst'] == True)[0]].copy()
    else:
        df_filt = df_features.copy()

    # Get periods
    periods = df_filt['period'].values / fs

    # Convert periods to freqs
    freqs = 1 / periods
    df_filt['freqs'] = freqs

    # Get cycles within range
    cycles = np.where((freqs >= f_range[0]) & (freqs < f_range[1]))[0]
    df_filt = df_filt.iloc[cycles]

    # Return nan if no cycles are found within freq range
    if len(df_filt) == 0:

        return np.nan

    return df_filt
