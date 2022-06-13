"""Optimize spectral fits."""

import warnings

import numpy as np
from scipy.optimize import curve_fit

from neurodsp.spectral import compute_spectrum

from fooof import FOOOF
from fooof.core.funcs import gaussian_function
from fooof.utils.params import compute_gauss_std
from fooof.sim.gen import gen_periodic, gen_aperiodic

from .emd import compute_emd, limit_freqs_hht, compute_it_emd


def refit(fm, sig, fs, f_range, emd_method='original', emd_kwargs=None,
          power_thresh=.2, energy_thresh=0., refit_ap=False):
    """Refit a power spectrum using EMD based parameter estimation.

    Parameters
    ----------
    fm : fooof.FOOOF
        FOOOF object containing results from fitting.
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of [float, float]
        Frequency range to restrict power spectrum to.
    emd_method : {'original', 'iterative'}
        EMD method type.
    emd_kwargs : optional, default: None
        Optional keyword arguments for compute_emd. Includes:

        - max_imfs
        - sift_thresh
        - env_step_size
        - max_iters
        - energy_thresh
        - stop_method
        - sd_thresh
        - rilling_thresh

    power_thresh : float, optional, default: .2
        IMF power threshold as the mean power above the initial aperiodic fit.
    energy_thresh : float, optional, default: 0.
        Normalized HHT energy threshold to define oscillatory frequencies. This aids the removal of
        harmonic peaks if present.
    refit_ap : bool, optional, default: None
        Refits the aperiodic component when True. When False, the aperiodic component is defined
        from the intial specparam fit.

    Returns
    -------
    fm : fooof.FOOOF
        Updated FOOOF fit.
    imf : 2d array
        Intrinsic modes functions.
    pe_mask : 1d array
        Booleans to mark imfs above aperiodic fit.
    """

    fm_refit = fm.copy()

    if emd_kwargs is None and emd_method == 'original':
        emd_kwargs = {'sd_thresh': .1}
    elif emd_kwargs is None:
        emd_kwargs = {}

    # Compute modes
    if emd_method == 'original':
        imf = compute_emd(sig, **emd_kwargs)
    else:
        imf = compute_it_emd(sig, fs, **emd_kwargs)

    # Convert spectra of mode timeseries
    _, powers_imf = compute_spectrum(imf, fs, f_range=f_range)

    freqs = fm_refit.freqs
    powers = fm_refit.power_spectrum
    powers_imf = np.log10(powers_imf)

    # Initial aperiodic fit
    powers_ap = fm_refit._ap_fit

    # Select superthreshold modes
    pe_mask = select_modes(powers_imf, powers_ap, power_thresh=power_thresh)

    # Refit periodic
    if not pe_mask.any():
        warnings.warn('No IMFs are above the intial aperiodic fit. '
                      'Returning the inital spectral fit.')
        return fm_refit, imf, pe_mask

    if energy_thresh > 0:

        # Limit frequency ranges to fit using HHT
        freqs_min, freqs_max = limit_freqs_hht(imf[pe_mask], freqs, fs,
                                               energy_thresh=energy_thresh)

        if freqs_min is None and freqs_max is None:
            warnings.warn('No superthreshold energy in HHT. '
                          'Returning the inital spectral fit.')
            return fm_refit, imf, np.zeros(len(pe_mask), dtype=bool)

        limits = (freqs_min, freqs_max)

        gauss_params = fit_gaussians(freqs, powers, powers_imf, powers_ap, pe_mask, limits)

    else:

        gauss_params = fit_gaussians(freqs, powers, powers_imf, powers_ap, pe_mask)

    if gauss_params is not None:

        fm_refit.peak_params_ = fm_refit._create_peak_params(gauss_params)
        fm_refit._peak_fit = gen_periodic(freqs, gauss_params.flatten())

    else:
        fm_refit.peak_params_ = None
        fm_refit._peak_fit = np.zeros_like(freqs)

    # Refit aperiodic
    if refit_ap:
        ap_params, ap_fit = refit_aperiodic(freqs, powers, fm_refit._peak_fit)
        fm_refit._ap_fit = ap_fit
        fm_refit.aperiodic_params_ = ap_params

    # Update attibutes
    fm_refit.gaussian_params_ = gauss_params
    fm_refit.fooofed_spectrum_ = fm_refit._peak_fit + fm_refit._ap_fit

    fm_refit._calc_r_squared()
    fm_refit._calc_error()

    return fm_refit, imf, pe_mask


def select_modes(powers_imf, powers_ap, power_thresh=0.2):
    """Get indices of modes that are over the initial aperiodic fit.

    Parameters
    ----------
    powers_imf : 2d array
        Modes spectral power, in log10 scale.
    powers_ap : 1d array
        Aperiodic spectral power, in log10 scale.
    power_thresh : float, optional, default: 0.2
        Mean power above the aperiodic threshold.

    Returns
    -------
    pe_mask : 1d array
        Booleans to mark imfs above aperiodic fit.
    """

    power_over_ap = np.zeros(len(powers_imf))

    for ind, power_imf in enumerate(powers_imf):

        diff = power_imf - powers_ap
        inds = np.where(diff > 0)[0]

        if len(inds) > 2:
            power_over_ap[ind] = np.mean(diff[inds])

    pe_mask = power_over_ap > power_thresh

    return pe_mask


def guess_params(freqs, powers, power_imf, ap_fit, inds):
    """Intial parameters esimates.

    Parameters
    ----------
    freqs : 1d array
        Frequency values for the power spectrum, in linear scale.
    powers : 1d array
        Power values, in log10 scale.
    power_imf : 1d array
        Power values of an intrisic mode.
    ap_fit : 1d array
        Power values of the inital aperiodic fit.
    inds : 1d array
        Indices to limit the range of the fit.

    Returns
    -------
    guess : list
        Center, height, and width estimates, respectively.
    bounds : list of tuple
        Lower and upper parameter bounds.
    """
    # Remove aperiodic fit from imf (i.e. flatten)
    power_imf = power_imf - ap_fit

    # Estimate center location and power height
    center = np.argmax(powers[inds])
    height = powers[inds][center] - ap_fit[inds][center]

    # Widths
    if center == 0 or center == len(inds):
        fwhm = (freqs[-1] - freqs[0]) / 2
        min_fwhm = .25 * fwhm
        max_fwhm = 2 * fwhm

    else:

        right = power_imf[inds][center:]
        left = power_imf[inds][:center]

        fwhm = freqs[inds[np.argmin(np.abs(right - (height * .5))) + center]] - \
            freqs[inds[np.argmin(np.abs(left - (height * .5)))]]

        min_fwhm = freqs[inds[np.argmin(np.abs(right - (height * .6))) + center]] - \
            freqs[inds[np.argmin(np.abs(left - (height * .6)))]]

        max_fwhm = freqs[inds[np.argmin(np.abs(right - (height * .4))) + center]] - \
            freqs[inds[np.argmin(np.abs(left - (height * .4)))]]

    # Convert fwhm to std
    width = compute_gauss_std(fwhm)
    min_width = compute_gauss_std(min_fwhm)
    max_width = compute_gauss_std(max_fwhm)

    # Non-monotonic or short sequences may produce infesible width bounds
    if min_width > width:
        min_width = width * .5

    if max_width < width:
        max_width = width * 2

    if min_width == max_width:
        min_width = width * .5
        max_width = width * 2

    # Convert center to frequency
    center = freqs[inds][center]

    # Estimate parameters
    guess = [center, height, width]

    # Estimate bounds
    min_center = center - width
    max_center = center + width

    lower_bounds = [min_center, height - 2*np.std(powers[inds]), min_width]
    upper_bounds = [max_center, height + 2*np.std(powers[inds]), max_width]

    # Negative bounds to zero
    lower_bounds = [0 if bound < 0 else bound for bound in lower_bounds]
    upper_bounds = [0 if bound < 0 else bound for bound in upper_bounds]

    bounds = [lower_bounds, upper_bounds]

    return guess, bounds


def fit_gaussians(freqs, powers, powers_imf, powers_ap, pe_mask, limits=None):
    """Fit gaussians based on EMD estimation.

    Parameters
    ----------
    freqs : 1d array
        Frequency values for the power spectrum, in linear scale.
    powers : 1d array
        Power values, in log10 scale.
    power_imf : 1d array
        Power values of an intrisic mode.
    powers_ap : 1d array
        Aperiodic spectral power, in log10 scale.
    pe_mask : 1d array
        Booleans to mark imfs above aperiodic fit.
    limits : list of tuple
        Lower and upper frequency bounds to limit parameter estimation to.

    Returns
    -------
    gauss_params : 2d array
        Gaussian parameters fits with shape (n_gaussians, [center, height, width]).
    """
    # Guess fit parameters
    guess = []

    # Upper and lower bounds
    bounds = [[], []]

    # Remove frequencies out of HHT bounds to account for asym harmonics
    if limits is not None:

        _freqs = np.array([], dtype=int)
        for lower, upper in zip(*limits):

            inds = np.arange(lower, upper, dtype='int')

            if len(inds) < 3:
                continue

            _guess, _bounds = guess_params(freqs, powers, powers, powers_ap, inds)

            guess.extend(_guess)

            bounds[0].extend(_bounds[0])
            bounds[1].extend(_bounds[1])

    else:

        for power_imf in powers_imf[pe_mask]:

            inds = np.where(power_imf > powers_ap)[0]

            # Ensure indices are continuous
            inds = np.arange(min(inds), max(inds)+1)

            _guess, _bounds = guess_params(freqs, powers, power_imf, powers_ap, inds)

            guess.extend(_guess)

            bounds[0].extend(_bounds[0])
            bounds[1].extend(_bounds[1])

    # Nothing to fit
    if len(guess) == 0:
        return None

    # Fit the flatten spectrum
    spectrum_flat = powers - powers_ap

    gauss_params, _ = curve_fit(gaussian_function, freqs, spectrum_flat, p0=guess, bounds=bounds)

    gauss_params = gauss_params.reshape(-1, 3)

    return gauss_params


def refit_aperiodic(freqs, powers, peak_fit):
    """Refit the aperiodic component following the periodic refit.

    Parameters
    ----------
    freqs : 1d array
        Frequency values for the power spectrum, in linear scale.
    powers : 1d array
        Power values, in log10 scale.
    peak_fit : 1d array
        Perodic refit, in log10 scale.

    Returns
    -------
    ap_params : 1d array
        Exponent and offset values.
    ap_fit : 1d array
        Regenerated aperiodic fit.
    """
    # Access simple ap fit method
    _fm = FOOOF()
    _fm.power_spectrum = powers
    _fm.freqs = freqs

    # Remove peaks
    spectrum_peak_rm = powers - peak_fit

    # Refit
    ap_params = _fm._simple_ap_fit(freqs, spectrum_peak_rm)

    ap_fit = gen_aperiodic(freqs, ap_params)

    return ap_params, ap_fit
