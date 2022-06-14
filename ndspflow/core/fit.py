"""FOOOF and Bycycle model fitting."""

import warnings

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


def fit_bycycle(sig, fs, f_range, center_extrema='peak', burst_method='cycles',
                threshold_kwargs=None, find_extrema_kwargs=None, round_samples=None,
                axis=0, verbose=False, n_jobs=-1):
    """A generalized Bycycle compute_features function to handle 1d, 2d, or 3d arrays.

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
    threshold_kwargs : dict, optional, default: None
        Feature thresholds for cycles to be considered bursts.
    find_extrema_kwargs : dict, optional, default: None
        Keyword arguments for function to find peaks an troughs (``find_extrema``)
        to change filter Parameters or boundary. By default, it sets the filter length to three
        cycles of the low cutoff frequency (``f_range[0]``).
    round_samples : int, optional, default: None
        Round samples of cyclepoints.
    n_jobs : int, optional, default: -1
        The number of jobs, one per cpu, to compute features in parallel.
    verbose : bool, optional, default: False
        Suppress warnings when False.

    Returns
    -------
    df_features : pandas.DataFrame
        A dataframe containing cycle features.

    Notes
    -----
    See bycycle documentation for more details.
    """

    if not verbose:
        warnings.simplefilter("ignore")

    threshold_kwargs = {} if not threshold_kwargs else threshold_kwargs
    find_extrema_kwargs = {} if not find_extrema_kwargs else find_extrema_kwargs

    compute_kwargs = dict(
        center_extrema=center_extrema, burst_method=burst_method,
        threshold_kwargs=threshold_kwargs, find_extrema_kwargs=find_extrema_kwargs
    )

    if sig.ndim == 1:

        df_features = compute_features(sig, fs, f_range, **compute_kwargs)

        if round_samples is not None:
            df_features['sample_last_trough'] = \
                df_features['sample_last_trough'].round(round_samples)

            df_features['sample_next_trough'] = \
                df_features['sample_next_trough'].round(round_samples)

            df_features['period'] = (
                (df_features['sample_next_trough'] -  df_features['sample_last_trough'])
            )

            df_features['freqs'] = fs/df_features['period']

    elif sig.ndim == 2:

        df_features = compute_features_2d(
            sig, fs, f_range, compute_features_kwargs=compute_kwargs,
            axis=axis, n_jobs=n_jobs
        )

    elif sig.ndim == 3:

        df_features = compute_features_3d(
            sig, fs, f_range, compute_features_kwargs=compute_kwargs,
            axis=axis, n_jobs=n_jobs
        )

    else:
        raise ValueError('The sig argument must specify a 1d, 2d, or 3d array.')



    return df_features
