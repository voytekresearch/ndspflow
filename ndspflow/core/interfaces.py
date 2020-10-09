""" Interface definitions."""

import os
import numpy as np
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    SimpleInterface,
    TraitedSpec,
    traits
)

from fooof import FOOOF, FOOOFGroup
from ndspflow.core.fit import fit_fooof, fit_bycycle
from ndspflow.io.save import save_fooof
from ndspflow.reports.html import generate_report


class FOOOFNodeInputSpec(BaseInterfaceInputSpec):
    """Input interface for FOOOF."""

    # Input/Output
    input_dir = traits.Directory(
        argstr='%s',
        exists=True,
        resolve=True,
        desc='Input directory containing timeseries and/or spectra .npy files to read.',
        mandatory=True,
        position=0
    )
    output_dir = traits.Directory(
        argstr='%s',
        exists=False,
        resolve=True,
        desc='Output directory to write results and BIDS derivatives to write.',
        mandatory=True,
        position=1
    )

    # Init params
    peak_width_limits = traits.Tuple((0.5, 12.0), mandatory=False, usedefault=True)
    max_n_peaks = traits.Int(100, mandatory=False, usedefault=True)
    min_peak_height = traits.Float(0.0, mandatory=False, usedefault=True)
    peak_threshold = traits.Float(2.0, mandatory=False, usedefault=True)
    aperiodic_mode = traits.Str('fixed', mandatory=False, usedefault=True)

    # Fit params
    freqs = traits.File(mandatory=True, usedefault=False)
    power_spectrum = traits.File(mandatory=True, usedefault=False)
    freq_range = traits.Tuple((-np.inf, np.inf), mandatory=False, usedefault=True)
    n_jobs = traits.Int(1, mandatory=False, usedefault=True)


class FOOOFNodeOutputSpec(TraitedSpec):
    """Output interface for FOOOF."""

    fm = traits.Any(mandatory=True)
    fm_results = traits.Directory(mandatory=True)


class FOOOFNode(SimpleInterface):
    """Interface wrapper for FOOOF."""

    input_spec = FOOOFNodeInputSpec
    output_spec = FOOOFNodeOutputSpec

    def _run_interface(self, runtime):

        freqs = np.load(os.path.join(os.getcwd(), self.inputs.input_dir, self.inputs.freqs))
        powers = np.load(os.path.join(self.inputs.input_dir, self.inputs.power_spectrum))

        init_kwargs = {'peak_width_limits': self.inputs.peak_width_limits,
                       'max_n_peaks': self.inputs.max_n_peaks,
                       'min_peak_height': self.inputs.min_peak_height,
                       'peak_threshold': self.inputs.peak_threshold,
                       'aperiodic_mode': self.inputs.aperiodic_mode,
                       'verbose': False}
        # Fit
        fms = fit_fooof(freqs, powers, self.inputs.freq_range, init_kwargs, self.inputs.n_jobs)

        # Save model
        save_fooof(fms, self.inputs.output_dir)

        # Save reports
        generate_report(self.inputs.output_dir, fms)

        self._results["fm"] = fms
        self._results["fm_results"] = os.path.join(self.inputs.output_dir, 'fooof')

        return runtime


class BycycleNodeInputSpec(BaseInterfaceInputSpec):
    """Input interface for bycycle."""

    # Input/Output
    input_dir = traits.Directory(
        argstr='%s',
        exists=True,
        resolve=True,
        desc='Input directory containing timeseries and/or spectra .npy files to read.',
        mandatory=True,
        position=0
    )
    output_dir = traits.Directory(
        argstr='%s',
        exists=False,
        resolve=True,
        desc='Output directory to write results and BIDS derivatives to write.',
        mandatory=True,
        position=1
    )

    # Required arguments
    sig = traits.File(mandatory=True, usedefault=False)
    fs = traits.Float(mandatory=True, usedefault=False)
    f_range = traits.Tuple(mandatory=True, usedefault=False)

    # Optional arguments
    center_extrema = traits.Str('peak', mandatory=False, usedefault=True)
    burst_method = traits.Str('cycles', mandatory=False, usedefault=True)
    burst_kwargs = traits.Dict({}, mandatory=False, usedefault=True)
    threshold_kwargs = traits.Dict({}, mandatory=False, usedefault=True)
    find_extrema_kwargs = traits.Dict({}, mandatory=False, usedefault=True)
    n_jobs = traits.Int(1, mandatory=False, usedefault=True)


class BycycleNodeOutputSpec(TraitedSpec):
    """Output interface for bycycle."""

    df_features = traits.Any(mandatory=True)
    df_samples = traits.Any(mandatory=True)
    bycycle_results = traits.Directory(mandatory=True)

class BycycleNode(SimpleInterface):
    """Interface wrapper for bycycle."""

    input_spec = FOOOFNodeInputSpec
    output_spec = FOOOFNodeOutputSpec

    def _run_interface(self, runtime):

        sig = np.load(os.path.join(os.getcwd(), self.inputs.input_dir, self.inputs.sig))

        fit_kwargs = dict(
            center_extrema=self.inputs.center_extrema, burst_method=self.inputs.burst_method,
            burst_kwargs=self.inputs.burst_kwargs, threshold_kwargs=self.inputs.threshold_kwargs,
            find_extrema_kwargs=self.inputs.find_extrema_kwargs, n_jobs=self.inputs.n_jobs
        )

        # Fit
        df_features, df_samples = fit_bycycle(sig, self.fs, self.f_range, **fit_kwargs)

        # Save dataframes
        save_bycycle(df_features, df_samples, output_dir)

        # Save reports
        settings = fit_kwargs.copy()
        settings['f_range'] = f_range

        # CREATING RESULTS STRINGS FOR BYCYCLE COULD BE HELPFUL HERE
        generate_report(self.inputs.output_dir, (df_features, df_samples, settings))

        self._results["df_features"] = df_features
        self._results["df_samples"] = df_samples
        self._results["bycycle_results"] = os.path.join(self.inputs.output_dir, 'bycycle')
