""" Interface definitions."""

import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
)


class _FOOOFInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for FOOOF"""

    # Init params
    peak_width_limits = traits.Tuple((0.5, 12.0), mandatory=False, usedefault=True)
    max_n_peaks = traits.Int(np.inf, mandatory=False, usedefault=True)
    min_peak_height = traits.Float(0.0, mandatory=False, usedefault=True)
    peak_threshold = traits.Int(2.0, mandatory=False, usedefault=True)
    periodic_mode = traits.Str('fixed', mandatory=False, usedefault=True)

    # Fit params
    freqs = traits.File(mandatory=True, usedefault=False)
    power_spectrum = traits.File(mandatory=True, usedefault=False)
    freq_range = traits.Tuple((-np.inf, np.inf), mandatory=False, usedefault=True)


class _FOOOFOutputSpec(TraitedSpec):
    """Output interface wrapper for FOOOF"""

    fm = traits.Any(mandatory=True)


class FOOOF(SimpleInterface):
    """Interface wrapper for FetchNodesLabels."""

    input_spec = _BycycleInputSpec
    output_spec = _BycycleOutputSpec

    def _run_interface(self, runtime):

        from fooof import FOOOF

        fm = FOOOF(peak_width_limits=self.inputs.peak_width_limits,
                   max_n_peaks=self.inputs.max_n_peaks,
                   min_peak_height=self.inputs.min_peak_height,
                   peak_threshold=self.inputs.peak_threshold,
                   aperiodic_mode=self.inputs.aperiodic_mode,
                   verbose=False)

        freq_range = None if self.inputs.freq_range == (-np.inf, np.inf) else self.inputs.freq_range

        freqs = np.load(self.inputs.freqs)
        power_spectrum = np.load(self.inputs.power_spectrum)

        fm.fit(freqs, power_spectrum, freq_range=freq_range)

        self._results["fm"] = fm
