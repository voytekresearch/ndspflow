========
ndspflow
========

|CircleCI|_ |Codecov|_

.. |CircleCI| image:: https://circleci.com/gh/voytekresearch/ndspflow.svg?style=svg&circle-token=b26555544cf83f79a4aa45f6f4b98423e2ee06d0
.. _CircleCI: https://circleci.com/gh/voytekresearch/ndspflow

.. |Codecov| image:: https://codecov.io/gh/voytekresearch/ndspflow/branch/master/graph/badge.svg?token=I9Z7OPIZ7J
.. _Codecov: https://codecov.io/gh/voytekresearch/ndspflow

A `nipype <https://github.com/nipy/nipype>`_ powered workflow for running a pre-processed neural timeseries
through either `FOOOF <https://github.com/fooof-tools/fooof>`_ and/or `bycycle <https://github.com/bycycle-tools/bycycle>`_.

Dependencies
------------

- nipype >= 1.5.1
- fooof >= 1.0.0
- bycycle >= 1.0.0rc2
- plotly >= 4.10.0
- pytest >= 6.0.2


Motivations
-----------

- A cloud-deployable BIDS application (i.e. the cli to facilitate HPC job submission).
- Interactive visualization reports (i.e. html pages to help assess group and individial fits).
- Containerization of workflows (i.e. docker and singularity for reproducibility).
- Facilitate cross bycycle/fooof workflows (i.e. using fooof for burst detection in bycycle).


Installation
------------

.. code-block:: shell

    $ git clone git@github.com:voytekresearch/ndspflow.git
    $ cd ndspflow
    $ pip install .

Usage
-----

.. code-block::

    $ ndspflow -h
        usage: ndspflow [-h] [-power_spectrum powers.npy] [-freqs freqs.npy] [-f_range_fooof lower_freq upper_freq] [-sig signal.npy] [-fs int]
                        [-f_range_bycycle lower_freq upper_freq] [-peak_width_limits lower_limit upper_limit] [-max_n_peaks int]
                        [-min_peak_height float] [-peak_threshold float] [-aperiodic_mode {fixed,knee}] [-center_extrema {peak,trough}]
                        [-burst_method {cycles,amp}] [-amp_fraction_threshold float] [-amp_consistency_threshold float]
                        [-period_consistency_threshold float] [-monotonicity_threshold float] [-min_n_cycles int] [-burst_fraction_threshold float]
                        [-axis {0, 1, 0, 1, None}] [-n_jobs int] [-run_nodes {fooof,bycycle}]
                        /path/to/input /path/to/output

        A Nipype workflow for FOOOOF and Bycycle.

        positional arguments:
        /path/to/input        Input directory containing timeseries and/or spectra .npy files to read (default: None).
        /path/to/output       Output directory to write results and BIDS derivatives to write (default: None).

        optional arguments:
        -power_spectrum powers.npy
                                Filename of power values, located inside of 'input_dir'
                                Required if 'fooof' in 'run_nodes argument' (default: None).
        -freqs freqs.npy      Filename of frequency values for the power spectrum(a), located inside of 'input_dir'.
                                Required if 'fooof' in 'run_nodes argument' (default: None).
        -f_range_fooof lower_freq upper_freq
                                Frequency range of the power spectrum, as: lower_freq, upper_freq.
                                Recommended if 'fooof' in 'run_nodes argument' (default: (-inf, inf)).
        -sig signal.npy       Filename of neural signal or timeseries, located inside of 'input_dir'.
                                Required if 'bycycle' in 'run_nodes argument' (default: None).
        -fs int               Sampling rate, in Hz.
                                Required if 'bycycle' in 'run_nodes argument'.
        -f_range_bycycle lower_freq upper_freq
                                Frequency range for narrowband signal of interest (Hz).
                                Required if 'bycycle' in 'run_nodes argument'.

        -peak_width_limits lower_limit upper_limit
                                Limits on possible peak width, in Hz, as: lower_limit upper_limit.
                                Recommended if 'fooof' in 'run_nodes argument' (default: (0.5, 12.0)).
        -max_n_peaks int      Maximum number of peaks to fit.
                                Recommended if 'fooof' in 'run_nodes argument' (default: 100).
        -min_peak_height float
                                Absolute threshold for detecting peaks, in units of the input data.
                                Recommended if 'fooof' in 'run_nodes argument' (default: 0.0).
        -peak_threshold float
                                Relative threshold for detecting peaks, in units of standard deviation of the input data.
                                Recommended if 'fooof' in 'run_nodes argument' (default: 2.0).
        -aperiodic_mode {fixed,knee}
                                Which approach to take for fitting the aperiodic component.
                                Recommended if 'fooof' in 'run_nodes argument' (default: fixed).

        -center_extrema {peak,trough}
                                Determines if cycles or peak or trough centered.
                                Recommended if 'bycycle' in 'run_nodes argument' (default: peak).
        -burst_method {cycles,amp}
                                Method for burst detection.
                                Recommended if 'bycycle' in 'run_nodes argument' (default: cycles).
        -amp_fraction_threshold float
                                Amplitude fraction threshold for detecting bursts.
                                Recommended if 'burst_method' is 'cycles' (default: 0).
        -amp_consistency_threshold float
                                Amplitude consistency threshold for detecting bursts.
                                Recommended if 'burst_method' is 'cycles' (default: 0.5).
        -period_consistency_threshold float
                                Period consistency threshold for detecting bursts.
                                Recommended if 'burst_method' is 'cycles' (default: 0.5).
        -monotonicity_threshold float
                                Monotonicicity threshold for detecting bursts.
                                Recommended if 'burst_method' is 'cycles' (default: 0.8).
        -min_n_cycles int     Minium number of cycles for detecting bursts
                                Recommended for either 'burst_method' (default: 3).
        -burst_fraction_threshold float
                                Minimum fraction of a cycle identified as a burst.
                                Recommended if 'burst_method' is 'amp' (default: 1).
        -axis {0, 1, (0, 1), None}
                                The axis to compute features across for 2D and 3D signal arrays.
                                Ignored if signal is 1D. 1 and (0, 1) only availble for 3D signals
                                (default: 0).

        -n_jobs int           The maximum number of jobs to run in parallel at one time.
                                Only utilized for 2d and 3d arrays (default: 1).
        -run_nodes {fooof,bycycle}
                                List of nodes to run: fooof and/or bycyle (default: fooof bycycle).

