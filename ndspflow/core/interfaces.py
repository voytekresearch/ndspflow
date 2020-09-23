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


class FOOOFInputSpec(BaseInterfaceInputSpec):
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


class FOOOFOutputSpec(TraitedSpec):
    """Output interface for FOOOF"""

    fm = traits.Any(mandatory=True)
    fm_results = traits.Directory(mandatory=True)


class FOOOF(SimpleInterface):
    """Interface wrapper for FOOOF."""

    input_spec = FOOOFInputSpec
    output_spec = FOOOFOutputSpec

    def _run_interface(self, runtime):

        from fooof import FOOOF

        fm = FOOOF(peak_width_limits=self.inputs.peak_width_limits,
                   max_n_peaks=self.inputs.max_n_peaks,
                   min_peak_height=self.inputs.min_peak_height,
                   peak_threshold=self.inputs.peak_threshold,
                   aperiodic_mode=self.inputs.aperiodic_mode,
                   verbose=False)

        freqs = np.load(os.path.join(os.getcwd(), self.inputs.input_dir, self.inputs.freqs))
        power_spectrum = np.load(os.path.join(self.inputs.input_dir, self.inputs.power_spectrum))

        fm.fit(freqs, power_spectrum, freq_range=self.inputs.freq_range)

        fm.save('fooof_results', file_path=self.inputs.output_dir, append=False,
                save_results=True, save_settings=True)

        self._results["fm"] = fm
        self._results["fm_results"] = os.path.join(self.inputs.output_dir, 'fooof_results')

        return runtime
