"""Motif class object."""

import matplotlib.pyplot as plt
import numpy as np

from neurodsp.plts import plot_time_series, plot_power_spectra
from neurodsp.spectral import compute_spectrum

from ndspflow.core.fit import fit_bycycle
from ndspflow.motif.burst import motif_burst_detection


class Motif:
    """Motif search and signal decomposition.

    Attributes
    ----------
    fm : fooof.FOOOF or list of tuple
        A fooof model that has been fit, or a list of (center_freq, bandwidth).
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    sig_pe : 2d array
        The reconstructed periodic signals. The zeroth index corresponds to frequency ranges.
    sig_ap : 2d array
        The reconstructed aperiodic signals. The zeroth index corresponds to frequency ranges.
    tforms : list of list of 2d array
        The affine matrix. Only returned when transform is True. The zeroth index corresponds to
        frequency ranges and the first index corresponds to individual cycles.
    corr_thresh : float, default: 0.5
        Correlation coefficient threshold.
    var_thresh : float, default: 0.05
        Height threshold in variance.
    min_clust_score : float, default: 1
        The minimum silhouette score to accept k clusters. The default skips clustering.
    min_clusters : int, default: 2
        The minimum number of clusters to evaluate.
    max_clusters : int, default: 10
        The maximum number of clusters to evaluate.
    min_n_cycles : int, odefault: 10
        The minimum number of cycles required to be considered at motif.
    center : str, {'peak', 'trough'}
        Center extrema definition.
    """


    def __init__(self, corr_thresh=0.5, var_thresh=0.05, min_clust_score=0.5,
                 min_clusters=2, max_clusters=10, min_n_cycles=10, center='peak'):
        """Initialize the object."""

        # Optional settings
        self.corr_thresh = corr_thresh
        self.var_thresh = var_thresh
        self.min_clust_score = min_clust_score
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_n_cycles = min_n_cycles
        self.center = center

        # Fit args
        self.fm = None
        self.sig = None
        self.fs = None

        # Results
        self.results = []
        self.sig_pe = None
        self.sig_ap = None
        self.tforms = None


    def __len__(self):
        """Define the length of the object."""

        return len(self.results)


    def __iter__(self):
        """Allow for iterating across the object."""

        for result in self.results:
            yield result


    def __getitem__(self, index):
        """Allow for indexing into the object."""

        return self.results[index]


    def fit(self, fm, sig, fs):
        """Robust motif extraction.

        Parameters
        ----------
        fm : fooof.FOOOF or list of tuple
        A fooof model that has been fit, or a list of (center_freq, bandwidth).
        sig : 1d array
            Time series.
        fs : float
            Sampling rate, in Hz.
        """

        from ndspflow.motif import extract

        self.fm = fm
        self.sig = sig
        self.fs = fs
        self.results = []

        # First pass motif extraction
        _motifs, _cycles = extract(self.fm, self.sig, self.fs, only_bursts=False,
                                   center=self.center, min_clusters=self.min_clusters,
                                   max_clusters=self.max_clusters)

        for motif, f_range in zip(_motifs, _cycles['f_ranges']):

            # Skip null motifs (np.nan)
            if isinstance(f_range, float):
                self.results.append(MotifResult(f_range))
                continue

            # Motif correlation burst detection
            bm = fit_bycycle(sig, fs, f_range)
            is_burst = motif_burst_detection(motif, bm, sig, corr_thresh=self.corr_thresh,
                                             var_thresh=self.var_thresh)
            bm['is_burst'] = is_burst

            # Re-extract motifs from bursts
            extract_kwargs = dict(
                center=self.center, only_bursts=True, var_thresh=self.var_thresh,
                min_clust_score=self.min_clust_score, min_clusters=self.min_clusters,
                max_clusters=self.max_clusters, min_n_cycles=self.min_n_cycles
            )

            motifs_burst, cycles_burst = extract(fm, sig, fs, df_features=bm, **extract_kwargs)

            # Match re-extraction results to frequency range of interest
            motif_idx = [idx for idx, cyc_range in enumerate(cycles_burst['f_ranges']) \
                        if not isinstance(cyc_range, float) and \
                        round(cyc_range[0] - f_range[0]) == 0 and \
                        round(cyc_range[1] - f_range[1]) == 0]

            # No cycles found in the given frequency range
            if len(motif_idx) != 1:
                self.results.append(MotifResult(f_range))
                continue

            motif_idx = motif_idx[0]

            # Collect results
            result = MotifResult(f_range, motifs_burst[motif_idx], cycles_burst['sigs'][motif_idx],
                                 cycles_burst['dfs_features'][motif_idx],
                                 cycles_burst['labels'][motif_idx])
            self.results.append(result)


    def decompose(self, center='peak', mean_center=True, transform=True):
        """Decompose a signal into its periodic/aperioidic components.

        Parameters
        ----------
        center : str, optional, {'peak', 'trough'}
            Center extrema definition.
        mean_center : bool, optional, default: True
            Global detrending (mean centering of the original signal).
        transfrom : bool, optional, default: True
            Applies an affine transfrom from motif to cycle if True.
        """

        if len(self.results) == 0:
            raise ValueError("Object must be fit prior to decomposing.")

        from ndspflow.motif import extract, decompose

        motifs = [result.motif for result in self.results]
        dfs_features = [result.df_features for result in self.results]
        labels = [result.labels for result in self.results]

        if transform:
            sigs_pe, sigs_ap, tforms = decompose(self.sig, motifs, dfs_features, center, labels,
                                                 mean_center, transform)
        else:
            sigs_pe, sigs_ap = decompose(self.sig, motifs, dfs_features, center, labels,
                                         mean_center, transform)

        sig_idx = 0
        for result in self.results:
            if not isinstance(result.motif, float):

                if transform:
                    result.add_decompose(sigs_pe[sig_idx], sigs_ap[sig_idx], tforms[sig_idx])
                else:
                    result.add_decompose(sigs_pe[sig_idx], sigs_ap[sig_idx])

                sig_idx += 1


    def plot(self, n_bursts=5, center='peak', normalize=True, plot_fm_kwargs=None, show=True):
        """Plot a motif summary.

        Parameters
        ----------
        n_bursts : int, optional, default: 5
            The number of example bursts to plot per peak.
        center : {'peak', 'trough'}, optional
            Defines centers of bycycle cycles.
        normalize : book, optiona, default: True
            Signal is mean centered with variance of one if True.
        plot_fm_kwargs : dict, optional, default: None
            Keyword arguments for the :func:`~.plot_fm` function.
        """

        from ndspflow.plts.motif import plot_motifs

        motifs = [result.motif for result in self.results]
        dfs_features = [result.df_features for result in self.results]

        fig = plot_motifs(self.fm, motifs, {'dfs_features': dfs_features}, self.sig, self.fs,
                          n_bursts, center, normalize, plot_fm_kwargs)

        if show:
            fig.show()


    def plot_decompose(self, result_index, **kwargs):
        """Plot periodic and aperiodic signal decomposition."""

        if self.results[result_index].sig_pe is None or self.results[result_index].sig_ap is None:
            self.decompose()

        times = np.arange(0, len(self.sig)/self.fs, 1/self.fs)

        figsize = kwargs.pop('figsize', (15, 2))
        alpha = kwargs.pop('alpha', [0.75, 1])

        # Plot the periodic decomposition
        plot_time_series(times, [self.sig, self.results[result_index].sig_pe],
                         labels=['Original', 'Periodic'], title='Periodic Reconstruction',
                         alpha=alpha, figsize=figsize, **kwargs)

        # Plot the aperiodic decomposition
        plot_time_series(times, [self.sig, self.results[result_index].sig_ap],
                         labels=['Original', 'Aperiodic'], title='Aperiodic Reconstruction',
                         alpha=alpha, figsize=figsize, **kwargs)


    def plot_spectra(self, result_index, f_range=(1, 100), figsize=(8, 8)):

        if self.results[result_index].sig_pe is None or self.results[result_index].sig_ap is None:
            self.decompose()

        # Compute spectra
        freqs, powers = compute_spectrum(self.sig, self.fs, f_range=f_range)

        freqs_pe, powers_pe = compute_spectrum(self.results[result_index].sig_pe,
                                               self.fs, f_range=f_range)

        freqs_ap, powers_ap = compute_spectrum(self.results[result_index].sig_ap,
                                               self.fs, f_range=f_range)

        # Plot
        _, ax = plt.subplots(figsize=figsize)

        plot_power_spectra(freqs, [powers, powers_pe, powers_ap],
                           title="Reconstructed Components",
                           labels=['Orig', 'PE Recon', 'AP Recon'],
                           ax=ax, alpha=[0.7, 0.7, 0.7], lw=3)

    def plot_transform(self, result_index, center='peak', xlim=None, figsize=(10, 1.5)):

        if self.results[result_index].sig_pe is None or self.results[result_index].sig_ap is None:
            self.decompose()

        fig, axes = plt.subplots(figsize=(figsize[0], figsize[1]*(7)), nrows=7, sharex=True)

        side = 'trough' if center == 'peak' else 'peak'
        starts = self.results[result_index].df_features['sample_last_' + side]
        ends = self.results[result_index].df_features['sample_next_' + side]

        # Get affine matrix parameters
        tforms = self.results[result_index].tforms
        rotations = [tform.rotation for tform in tforms]
        translations_x = [tform.translation[0] for tform in tforms]
        translations_y = [tform.translation[1] for tform in tforms]
        scales_x = [tform.scale[0] for tform in tforms]
        scales_y = [tform.scale[0] for tform in tforms]
        shears = [tform.shear for tform in tforms]

        params = [rotations, translations_x, translations_y, scales_x, scales_y, shears]

        param_arr = np.zeros((len(params), len(self.sig)))
        param_arr[:, :] = np.nan

        for idx, params in enumerate(params):
            for cyc_idx, (start, end) in enumerate(zip(starts, ends)):
                param_arr[idx][start:end] = params[cyc_idx]

        # Plot bursts
        times = np.arange(0, len(self.sig)/self.fs, 1/self.fs)
        plot_time_series(times, [self.sig, self.results[result_index].sig_pe], xlim=xlim,
                         alpha=[0.5, 1, 1], ax=axes[0])

        axes[0].set_ylabel("")
        axes[0].set_xlabel("")
        axes[0].set_title('Motif Detection and Fitting', fontsize=20)
        axes[0].tick_params(axis='y', labelsize=12)

        # Plot affine matrix params
        titles = ['Rotation', 'Translation X', 'Translation Y', 'Scale X', 'Scale Y', 'Shear']
        for idx in range(len(param_arr)):
            axes[idx+1].plot(times, param_arr[idx])
            axes[idx+1].set_xlim(xlim)
            axes[idx+1].set_title(titles[idx], fontsize=16)

        axes[len(param_arr)].set_xlabel('Time', fontsize=16)


class MotifResult:
    """Convenience class to access individual results by attibute.

    Attributes
    ----------
    f_range : tuple, optional, default: np.nan
        The frequency range requested or defined by a fooof fit.
    motif : list of 1d array
        Extracted motifs.
    sigs : 2d array, optional, default: np.nan
        Cycles resample to the center frequency of ``f_range``.
    df_features : pd.DataFrame, optional, default: np.nan
        Bycycle dataframe.
    labels : 1d array, optional, default: np.nan
        Cluster labels.
    sig_pe : 1d array
        Periodic signal.
    sig_ap : 1d array
        Aperiodic signal.
    tforms : list of 2d array, optional
        Transformation matrix for each cycle
    """

    def __init__(self, f_range, motif=np.nan, sigs=np.nan, df_features=np.nan, labels=np.nan):
        """Initialize object."""

        self.f_range = f_range
        self.motif = motif
        self.sigs = sigs
        self.df_features = df_features
        self.labels = labels

        self.sig_pe = None
        self.sig_ap = None
        self.tforms = None


    def add_decompose(self, sig_pe, sig_ap, tforms=None):
        """Add decomposed periodic and aperiodic signals.

        Parameters
        ----------
        sig_pe : 1d array
            Periodic signal.
        sig_ap : 1d array
            Aperiodic signal.
        tforms : list of 2d array, optional, default: None
            Transformation matrix for each cycle
        """

        self.sig_pe = sig_pe
        self.sig_ap = sig_ap
        self.tforms = tforms
