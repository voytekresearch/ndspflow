"""FOOOF and Bycycle model fitting."""

import os
from fooof import FOOOF, FOOOFGroup, fit_fooof_3d
from bycycle.features import compute_features
from bycycle.group import compute_features_2d, compute_features_3d


def fit_fooof(freqs, powers, freq_range, init_kwargs, n_jobs):
    """A generalized FOOOF fit function to handle 1d, 2d, or 3d arrays.

    Parameters
    ----------
    powers : 1d, 2d, or 3d array
        Power values for a single spectrum, or group of power spectra.
        Power values are stored internally in log10 scale.
    freqs : 1d array
        Frequency values for the power spectra.
    freq_range : list of [float, float]
        Frequency range of the power spectra, as [lowest_freq, highest_freq].
    init_kwargs : dict
        FOOOF object initialization kwargs.
    n_jobs : int
        Specificy the number of jobs to run in parrallel for 2d or 3d arrays.

    Returns
    -------
    model : FOOOF, FOOOFGroup, or list of FOOOFGroup objects.
        A FOOOF object that has been fit. A 1d array will return a FOOOF objects, a 2d array will
        return a FOOOFGroup object, and a 3d array will return a list of FOOOFGroup objects.
    """

    if powers.ndim == 1:
        # Fit a 1d array
        model = FOOOF(**init_kwargs)
        model.fit(freqs, powers, freq_range=freq_range)

    elif powers.ndim == 2:
        # Fit a 2d array
        model = FOOOFGroup(**init_kwargs)
        model.fit(freqs, powers, freq_range=freq_range, n_jobs=n_jobs)

    elif powers.ndim == 3:
        # Fit a 3d array
        model = FOOOFGroup(**init_kwargs)
        model = fit_fooof_3d(model, freqs, powers, n_jobs=n_jobs)

    else:
        raise ValueError('The power_spectrum argument must specify a 1d, 2d, or 3d array.')

    return model


def fit_bycycle(sig, fs, f_range, center_extrema='peak', burst_method='cycles', burst_kwargs=None,
                threshold_kwargs=None, find_extrema_kwargs=None, return_samples=True, n_jobs=1):
    """A generalized bycycle compute_features function to handle 1d, 2d, or 3d arrays.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    center_extrema : {'peak', 'trough'}
        The center extrema in the cycle.
    burst_method : string, optional, default: 'cycles'
        Method for detecting bursts.
    burst_kwargs : dict, optional, default: None
        Additional keyword arguments defined in :func:`~.compute_burst_fraction` for dual
        amplitude threshold burst detection (i.e. when burst_method == 'amp').
    threshold_kwargs : dict, optional, default: None
        Feature thresholds for cycles to be considered bursts.
    find_extrema_kwargs : dict, optional, default: None
        Keyword arguments for function to find peaks an troughs (``find_extrema``)
        to change filter Parameters or boundary. By default, it sets the filter length to three
        cycles of the low cutoff frequency (``f_range[0]``).
    return_samples : bool, optional, default: True
        Returns samples indices of cyclepoints used for determining features if True.
    n_jobs : int, optional, default: -1
        The number of jobs, one per cpu, to compute features in parallel.

    Returns
    -------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle.
    df_samples : pandas.DataFrame, optional, default: True
        An optionally returned dataframe containing cyclepoints for each cycle.

    Notes
    -----
    See bycycle documentation for more details.
    """

    compute_kwargs = dict(
        center_extrema=center_extrema, burst_method=burst_method, burst_kwargs=burst_kwargs,
        threshold_kwargs=threshold_kwargs, find_extrema_kwargs=find_extrema_kwargs,
        return_samples=return_samples
    )

    if sig.ndim == 1:

        df_features, df_samples = compute_features(sig, fs, f_range, **compute_kwargs)

    elif sig.ndim == 2:

        return_samples = compute_kwargs.pop(return_samples)

        df_features, df_samples = compute_features_2d(
            sig, fs, f_range, compute_features_kwargs=compute_kwargs,
            return_samples=return_samples, n_jobs=n_jobs, progress=None
        )

    elif sig.ndim == 3:

        return_samples = compute_kwargs.pop(return_samples)

        df_features, df_samples = compute_features_2d(
            sig, fs, f_range, compute_features_kwargs=compute_kwargs,
            return_samples=return_samples, n_jobs=n_jobs, progress=None
        )

    else:
        raise ValueError('The sig argument must specify a 1d, 2d, or 3d array.')

    return df_features, df_samples
