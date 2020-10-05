"""FOOOF and Bycycle model fitting."""

import os
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
    fm_labels : list of str
        Spectrum identifiers.
    """

    # Flatten the models and output dirs into a 1d list
    fms = []
    fm_paths = []
    fm_labels = []

    if type(model) is FOOOF:

        # For 1d arrays
        fm_paths.append(output_dir)
        fms.append(model)
        fm_labels.append("spectrum_{fm_idx}".format(fm_idx=str(0).zfill(4)))

    elif type(model) is FOOOFGroup:

        # For 2d arrays
        label_template = "spectrum_dim1-{dim_a}"
        for fm_idx in range(len(model)):

            label = label_template.format(dim_a=str(fm_idx).zfill(4))
            fm_labels.append(label)

            fm_paths.append(os.path.join(output_dir, label))
            fms.append(model.get_fooof(fm_idx))

    elif type(model) is list:

        # For 3d arrays
        label_template = "spectrum_dim1-{dim_a}_dim2-{dim_b}"

        for fg_idx in range(len(model)):

            for fm_idx in range(len(model[0].get_results())):

                label = label_template.format(dim_a=str(fg_idx).zfill(4),
                                              dim_b=str(fm_idx).zfill(4))
                fm_labels.append(label)
                fm_paths.append(os.path.join(output_dir, label))
                fms.append(model[fg_idx].get_fooof(fm_idx))

    return fms, fm_paths, fm_labels
