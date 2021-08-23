"""Optimize a group of spectral fits."""

from multiprocessing import Pool, cpu_count
from functools import partial

from bycycle.group.utils import progress_bar
from fooof.objs.utils import combine_fooofs

from .refit import refit


def refit_group(fg, sigs, fs, f_range, imf_kwargs=None, power_thresh=.2,
                 energy_thresh=0., refit_ap=False, n_jobs=-1, progress=None):
    """Refit a group of spectral fits.

    Parameters
    ----------
    fg : fooof.FOOOFGroup
        FOOOFGroup object containing results from fitting.
    sigs : 2d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of [float, float]
        Frequency range to restrict power spectrum to.
    imf_kwargs : optional, default: None
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
    fg_refit : fooof.FOOOFGroup
        FOOOFGroup object refit to IMF frequency ranges.
    imfs : 2d array
        Intrinsic mode function timeseries.
    pe_masks : 1d array
        Boolean array marking which imfs where used during refitting.
    """

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    if imf_kwargs is None:
        imf_kwargs = {'sd_thresh': .1}

    # Convert FOOOFGroup to list of FOOOF
    fms = [fg.get_fooof(ind) for ind in range(len(fg))]

    with Pool(processes=n_jobs) as pool:

        mapping = pool.imap(partial(_proxy, fs=fs, f_range=f_range, imf_kwargs=imf_kwargs,
                                    power_thresh=power_thresh, energy_thresh=energy_thresh,
                                    refit_ap=refit_ap),
                            zip(fms, sigs))

        results = list(progress_bar(mapping, progress, len(sigs), pbar_desc='Refitting Spectra'))

    fms = [result[0] for result in results]
    imfs= [result[1] for result in results]
    pe_masks = [result[2] for result in results]

    # Rebuild a group object
    fg_refit = combine_fooofs(fms)

    return fg_refit, imfs, pe_masks


def _proxy(args, fs=None, f_range=None, imf_kwargs=None,
           power_thresh=None, energy_thresh=None, refit_ap=None):

    fm, sig = args[0], args[1]

    return refit(fm, sig, fs, f_range, imf_kwargs, power_thresh, energy_thresh, refit_ap)
