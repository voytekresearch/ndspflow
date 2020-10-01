"""FOOOF and Bycycle model fitting."""

from fooof import FOOOF, FOOOFGroup, fit_fooof_3d


def fit_fooof(freqs, powers, freq_range, init_kwargs, n_jobs):
    """A generalized fit function to handle 1d, 2d, or 3d arrays.

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
