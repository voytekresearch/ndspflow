"""Interface definitions."""

import os
import numpy as np

from nipype.interfaces.base import BaseInterfaceInputSpec, SimpleInterface, TraitedSpec, traits

from ndspflow.core.fit import fit_fooof, fit_bycycle
from ndspflow.io.save import save_fooof, save_bycycle
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
    f_range_fooof = traits.Tuple((-np.inf, np.inf), mandatory=False, usedefault=True)
    n_jobs = traits.Int(1, mandatory=False, usedefault=True)


class FOOOFNodeOutputSpec(TraitedSpec):
    """Output interface for FOOOF."""

    fms = traits.Any(mandatory=True)
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
        fms = fit_fooof(freqs, powers, self.inputs.f_range_fooof, init_kwargs,
                        self.inputs.n_jobs)

        # Save model
        save_fooof(fms, self.inputs.output_dir)

        self._results["fms"] = fms
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
    f_range_bycycle = traits.Tuple(mandatory=True, usedefault=False)

    # Optional arguments
    center_extrema = traits.Str('peak', mandatory=False, usedefault=True)
    burst_method = traits.Str('cycles', mandatory=False, usedefault=True)
    amp_fraction_threshold = traits.Float(0.0, mandatory=False, usedefault=True)
    amp_consistency_threshold = traits.Float(0.5, mandatory=False, usedefault=True)
    period_consistency_threshold = traits.Float(0.5, mandatory=False, usedefault=True)
    monotonicity_threshold = traits.Float(0.8, mandatory=False, usedefault=True)
    min_n_cycles = traits.Int(3, mandatory=False, usedefault=True)
    burst_fraction_threshold = traits.Float(1.0, mandatory=False, usedefault=True)
    axis = traits.Str('None', mandatory=False, usedefault=True)
    n_jobs = traits.Int(1, mandatory=False, usedefault=True)


class BycycleNodeOutputSpec(TraitedSpec):
    """Output interface for bycycle."""

    df_features = traits.Any(mandatory=True)
    bm_results = traits.Directory(mandatory=True)
    _fit_args = traits.Any(mandatory=True)


class BycycleNode(SimpleInterface):
    """Interface wrapper for bycycle."""

    input_spec = BycycleNodeInputSpec
    output_spec = BycycleNodeOutputSpec

    def _run_interface(self, runtime):

        sig = np.load(os.path.join(os.getcwd(), self.inputs.input_dir, self.inputs.sig))

        # Infer axis type from string (traits doesn't support multi-type)
        axis = None  if 'None' in self.inputs.axis else self.inputs.axis
        axis = (0, 1) if self.inputs.axis.replace(' ', '') == '(0,1)' else axis

        axis_error = ValueError("Axis must be 0, 1, (0, 1), or None.")
        if axis is not None and axis != (0, 1):
            try:
                axis = int(self.inputs.axis)
            except:
                raise ValueError from axis_error

        if axis not in [0, 1, (0, 1), None]:
            raise ValueError from axis_error

        # Get thresholds
        if self.inputs.burst_method == 'cycles':

            threshold_kwargs = dict(
                amp_fraction_threshold = self.inputs.amp_fraction_threshold,
                amp_consistency_threshold = self.inputs.amp_consistency_threshold,
                period_consistency_threshold = self.inputs.period_consistency_threshold,
                monotonicity_threshold = self.inputs.monotonicity_threshold,
                min_n_cycles = self.inputs.min_n_cycles
            )

        else:

            threshold_kwargs = dict(
                burst_fraction_threshold = self.inputs.burst_fraction_threshold,
                min_n_cycles = self.inputs.min_n_cycles
            )

        # Organize all kwargs
        fit_kwargs = dict(
            center_extrema=self.inputs.center_extrema, burst_method=self.inputs.burst_method,
            threshold_kwargs=threshold_kwargs, axis=axis, n_jobs=self.inputs.n_jobs
        )

        # Fit
        df_features = fit_bycycle(sig, self.inputs.fs, self.inputs.f_range_bycycle, **fit_kwargs)

        # Save dataframes
        save_bycycle(df_features, self.inputs.output_dir)

        fit_args = dict(sig=sig, fs=self.inputs.fs, f_range=self.inputs.f_range_bycycle,
                        **fit_kwargs)

        self._results["df_features"] = df_features
        self._results["bm_results"] = os.path.join(self.inputs.output_dir, 'bycycle')
        self._results["_fit_args"] = fit_args

        return runtime


class ReportNodeInputSpec(BaseInterfaceInputSpec):
    """Input interface for reporting."""

    output_dir = traits.Directory(
        argstr='%s',
        exists=False,
        resolve=True,
        desc='Output directory to write results and BIDS derivatives to write.',
        mandatory=True,
        position=1
    )

    fms = traits.Any()
    df_features = traits.Any()
    _fit_args = traits.Any()


class ReportNodeOutputSpec(BaseInterfaceInputSpec):
    """Output interface for reporting."""


class ReportNode(SimpleInterface):
    """Interface wrapper for reporting."""

    input_spec = ReportNodeInputSpec
    output_spec = ReportNodeOutputSpec

    def _run_interface(self, runtime):

        fms = None if self.inputs.fms is None else self.inputs.fms
        bms = None if self.inputs.df_features is None else \
            (self.inputs.df_features, self.inputs._fit_args)

        generate_report(self.inputs.output_dir, fms=fms, bms=bms)

        return runtime
