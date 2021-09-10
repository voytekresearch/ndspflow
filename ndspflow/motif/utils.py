"""Utitlity array functions for determining motifs."""


import numpy as np
from scipy.signal import resample

from neurodsp.utils.norm import normalize_sig

from skimage.transform import (estimate_transform, AffineTransform, EuclideanTransform,
    SimilarityTransform, ProjectiveTransform, PolynomialTransform)


def split_signal(df_osc, sig, normalize=True, center='peak', n_samples=None):
    """Split the signal using a bycycle dataframe.

    Parameters
    ----------
    df_osc : pandas.DataFrame
        Dataframe containing bycycle features, that has been limited to an oscillation frequency
        range of interest.
    sig : 1d array
        Time series.
    normalize : bool, optional, default: True
        Normalizes each cycle (mean centers with variance of one) when True.
    center : {'peak', 'trough'}, optional
        Center definition of cycles.
    n_samples : int, optional
        Number of samples to resample cycles to. If None, the mean of cycles is used.

    Returns
    -------
    sigs : 2d array
        Each cycle resampled to the mean of the dataframe.
    """

    # Get the start/end samples of each cycle
    side = 'trough' if center == 'peak' else 'peak'
    cyc_start = df_osc['sample_last_' + side].values
    cyc_end = df_osc['sample_next_' + side].values

    # Get the average number of samples
    if n_samples is None:
        n_samples = np.mean(df_osc['period'].values, dtype=int)

    sigs = np.zeros((len(df_osc), n_samples))

    # Slice cycles and resample to center frequency
    for idx, (start, end) in enumerate(zip(cyc_start, cyc_end)):

        if start < 0:
            continue

        sig_cyc = sig[start:end]
        sig_cyc = resample(sig_cyc, num=n_samples)

        if normalize:
            sig_cyc = normalize_sig(sig_cyc, mean=0)

        sigs[idx] = sig_cyc

    return sigs


def motif_to_cycle(motif, cycle, ttype='affine', fixed=True):
    """Affine transform motif to cycle.

    Parameters
    ----------
    motif : 1d array
        Mean waveform.
    cycle : 1d array
        Individual cycle waveform.
    ttype : {'euclidean', 'similarity', 'affine', 'projective', 'polynomial'}
        Transformation type.
    fixed : bool, optional, default: True
        Fixes the last and first points to ensure continuity between cycles if True.

    Returns
    -------
    motif_trans : 1d array
        Motif waveform tranformed to cycle space.
    tform : 2d array
        Transformation matrix.
    """

    _times = np.arange(0, len(motif))

    src = np.vstack((_times, motif)).T
    dst = np.vstack((_times, cycle)).T

    tform = estimate_transform(ttype, src, dst)

    # Select requested transformation type
    if ttype == 'affine':
        tfunc = AffineTransform
    elif ttype == 'similarity':
        tfunc = SimilarityTransform
    elif ttype == 'euclidean':
        tfunc = EuclideanTransform
    elif ttype == 'projective':
        tfunc = ProjectiveTransform
    elif ttype == 'polynomial':
        tfunc = PolynomialTransform

    motif_trans = tfunc(tform.params)(src).T[1]

    # Re-transform with fixed start/end points
    if False:

        # Determine rotation
        ba = np.array([len(cycle), motif_trans[-1] - cycle[0]])
        bc = np.array([len(cycle), cycle[-1] - cycle[0]])

        angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        rot = -np.arccos(angle) if cycle[-1] > motif_trans[-1] else np.arccos(angle)

        # Determine translation
        trans_y = cycle[0] - motif[0]

        tform.params[1, 2] = trans_y
        tform.params[1, 0] = rot

        motif_trans = tfunc(tform.params)(src).T[1]

    return motif_trans, tform
