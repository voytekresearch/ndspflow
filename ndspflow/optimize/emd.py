"""Emprical mode decomposition."""

import emd


def compute_emd(sig, max_imfs=None, sift_thresh=1e-08, env_step_size=1, max_iters=1000,
                energy_thresh=None, stop_method='sd', sd_thresh=.1,
                rilling_thresh=(0.05, 0.5, 0.05)):
    """Wrapper function to compute empricical modes.

    Parameters
    ----------
    sig : 1d array
        Time series.
    max_imfs : int, optional, default: None
        The maximum number of IMFs to compute.
    sift_thresh : float, optional, default: 1e-8
        The threshold at which the overall sifting process will stop.
    env_step_size : float
        Scaling of envelope prior to removal at each iteration of sift. The
        average of the upper and lower envelope is muliplied by this value
        before being subtracted from the data. Values should be between
        0 > x >= 1 (Default value = 1)
    max_iters : int > 0
        Maximum number of iterations to compute before throwing an error
    energy_thresh : float > 0
        Threshold for energy difference (in decibels) between IMF and residual
        to suggest stopping overall sift. (Default is None, recommended value is 50)
    stop_method : {'sd','rilling','fixed'}
        Flag indicating which metric to use to stop sifting and return an IMF.
    sd_thresh : float
        Used if 'stop_method' is 'sd'. The threshold at which the sift of each
        IMF will be stopped. (Default value = .1)
    rilling_thresh : tuple
        Used if 'stop_method' is 'rilling', needs to contain three values (sd1, sd2, alpha).
        An evaluation function (E) is defined by dividing the residual by the
        mode amplitude. The sift continues until E < sd1 for the fraction
        (1-alpha) of the data, and E < sd2 for the remainder.
        See section 3.2 of http://perso.ens-lyon.fr/patrick.flandrin/NSIP03.pdf

    Returns
    -------
    imf : 2d array
        Timeseries with shape (n_modes, n_timepoints), in increasing frequency.
    """

    imf_opts = dict(env_step_size=env_step_size, max_iters=max_iters, energy_thresh=energy_thresh,
                    stop_method=stop_method, sd_thresh=sd_thresh, rilling_thresh=rilling_thresh)

    imf = emd.sift.sift(sig, imf_opts=imf_opts).T[::-1]

    return imf
