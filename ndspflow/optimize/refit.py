"""Optimize spectral fits."""

import numpy as np
from scipy.optimize import curve_fit

from .emd import compute_emd

from neurodsp.spectral import compute_spectrum

from fooof import FOOOF
from fooof.core.funcs import gaussian_function
from fooof.utils.params import compute_gauss_std
from fooof.sim.gen import gen_periodic, gen_aperiodic


def refit(fm, sig, fs, f_range, imf_kwargs={'sd_thresh': .1}, power_thresh=.2):
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
    imf_kwargs : optional, default: {'sd_thresh': .1}
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

    Returns
    -------
    fm : fooof.FOOOF
        Updated FOOOF fit.
    """
    # Compute modes
    imf = compute_emd(sig, **imf_kwargs)

    # Convert spectra of mode timeseries
    freqs_imf, powers_imf = compute_spectrum(imf, fs, f_range=f_range)

    # Convert to log-log
    freqs = fm.freqs
    powers = fm.power_spectrum
    powers_imf = np.log10(powers_imf)

    # Initial aperiodic fit
    powers_ap = fm._ap_fit

    # Select superthreshold modes
    pe_mask = select_modes(powers_imf, powers_ap, power_thresh=power_thresh)

    # Refit periodic
    gauss_params = fit_gaussians(freqs, powers, powers_imf, powers_ap, pe_mask)

    pe_fit = gen_periodic(freqs, gauss_params.flatten())
    pe_params = fm._create_peak_params(gauss_params)

    # Refit aperiodic
    ap_params, ap_fit = refit_ap(freqs, powers, pe_fit)

    # Update attibutes
    fm.gaussian_params_ = gauss_params

    fm._peak_fit = gen_periodic(freqs, gauss_params.flatten())
    fm.peak_params_ = pe_params

    fm._ap_fit = ap_fit
    fm.aperiodic_params_ = ap_params

    fm.fooofed_spectrum_ = fm._peak_fit + fm._ap_fit

    return fm


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

        if len(inds) != 0:
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
    center = np.argmax(power_imf[inds])
    height = powers[inds][center] - ap_fit[inds][center]

    # Estimate width
    fwhm = freqs[inds[np.argmin(np.abs(power_imf[inds][center:] - (height * .5))) + center]] - \
           freqs[inds[np.argmin(np.abs(power_imf[inds][:center] - (height * .5)))]]
    width = compute_gauss_std(fwhm)

    # Width bounds
    min_fwhm = freqs[inds[np.argmin(np.abs(power_imf[inds][center:] - (height * .6))) + center]] - \
               freqs[inds[np.argmin(np.abs(power_imf[inds][:center] - (height * .6)))]]
    min_width = compute_gauss_std(min_fwhm)

    max_fwhm = freqs[inds[np.argmin(np.abs(power_imf[inds][center:] - (height * .4))) + center]] - \
               freqs[inds[np.argmin(np.abs(power_imf[inds][:center] - (height * .4)))]]
    max_width = compute_gauss_std(max_fwhm)

    # Convert center to frequency
    center = freqs[inds][center]

    # Estimate parameters
    guess = [center, height, width]

    # Estimate bounds
    min_center = center - width
    max_center = center + width

    bounds = [
        (min_center, height - 2*np.std(powers[inds]), min_width),
        (max_center, height + 2*np.std(powers[inds]), max_width)
    ]

    return guess, bounds


def fit_gaussians(freqs, powers, powers_imf, powers_ap, pe_mask):
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

    Returns
    -------
    gauss_params : 2d array
        Gaussian parameters fits with shape (n_gaussians, [center, height, width]).
    """
    # Guess fit parameters
    guess = []

    # Upper and lower bounds
    bounds = [[], []]

    for ind, power_imf in enumerate(powers_imf[pe_mask][::-1]):

        inds = np.where(power_imf > powers_ap)[0]

        _guess, _bounds = guess_params(freqs, powers, power_imf, powers_ap, inds)

        guess.extend(_guess)

        bounds[0].extend(_bounds[0])
        bounds[1].extend(_bounds[1])

    # Fit the flatten spectrum
    spectrum_flat = powers - powers_ap

    gauss_params, _ = curve_fit(gaussian_function, freqs, spectrum_flat, p0=guess, bounds=bounds)

    gauss_params = gauss_params.reshape(-1, 3)

    return gauss_params


def refit_ap(freqs, powers, peak_fit):
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
