========
ndspflow
========

A `nipype <https://github.com/nipy/nipype>`_ powered workflow for running a pre-processed neural timeseries
through either `FOOOF <https://github.com/fooof-tools/fooof>`_ and/or `bycycle <https://github.com/bycycle-tools/bycycle>`_.

Dependencies
------------

- nipype >= 1.5.1
- fooof >= 1.0.0
- bycycle >=0.1.3
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
      usage: ndspflow [-h] [-power_spectrum powers.npy] [-freqs freqs.npy] [-freq_range lower_freq upper_freq]
                      [-peak_width_limits lower_limit upper_limit] [-max_n_peaks int] [-min_peak_height float] [-peak_threshold float]
                      [-periodic_mode {fixed,knee}] [-run_nodes {fooof,bycycle} [{fooof,bycycle} ...]]
                      /path/to/input /path/to/output

      A Nipype workflow for FOOOF and Bycycle.

      positional arguments:
      /path/to/input        Input directory containing timeseries and/or spectra .npy files to read (default: None).
      /path/to/output       Output directory to write results and BIDS derivatives to write (default: None).

      optional arguments:
      -h, --help            show this help message and exit
      -power_spectrum powers.npy
                              Filename of power values, located inside of 'input_dir'
                              Required if 'fooof' in 'run_nodes argument' (default: None).
      -freqs freqs.npy        Filename of frequency values for the power spectrum(a), located inside of 'input_dir'.
                              Required if 'fooof' in 'run_nodes argument' (default: None).
      -freq_range lower_freq upper_freq
                              Frequency range of the power spectrum, as: lower_freq, upper_freq.
                              Recommended if 'fooof' in 'run_nodes argument' (default: None).
      -peak_width_limits lower_limit upper_limit
                              Limits on possible peak width, in Hz, as: lower_limit upper_limit.
                              Recommended if 'fooof' in 'run_nodes argument' (default: [0.5, 12.0]).
      -max_n_peaks int        Maximum number of peaks to fit.
                              Recommended if 'fooof' in 'run_nodes argument' (default: inf).
      -min_peak_height float
                              Absolute threshold for detecting peaks, in units of the input data.
                              Recommended if 'fooof' in 'run_nodes argument' (default: 0.0).
      -peak_threshold float
                              Relative threshold for detecting peaks, in units of standard deviation of the input data.
                              Recommended if 'fooof' in 'run_nodes argument' (default: 2.0).
      -periodic_mode {fixed,knee}
                              Which approach to take for fitting the aperiodic component.
                              Recommended if 'fooof' in 'run_nodes argument' (default: fixed).
      -run_nodes {fooof,bycycle}
                              List of nodes to run: fooof and/or bycyle (default: fooof bycycle).
