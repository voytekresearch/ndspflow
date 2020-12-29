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

Quickstart
----------

The input data should be organized in the working directory and the directory and file names must
match the command-line call.

.. code-block::

    data
    ├── freqs.npy
    ├── powers.npy
    └── sigs.npy

The command-line call below runs both fooof and bycycle.

.. code-block::

    $ ndspflow \
      -freqs freqs.npy \
      -power_spectrum powers.npy \
      -sig sigs.npy \
      -fs 500 \
      -f_range_fooof 1 50 \
      -f_range_bycycle 15 25 \
      -max_n_peaks 1 \
      -min_peak_height .3 \
      -peak_threshold 2 \
      -peak_width_limits 1 5 \
      -aperiodic_mode fixed \
      -center_extrema peak \
      -burst_method cycles \
      -amp_fraction_threshold 0 \
      -amp_consistency_threshold .5 \
      -period_consistency_threshold .5 \
      -monotonicity_threshold .8 \
      -min_n_cycles 3 \
      -axis 0 \
      -n_jobs -1 \
      -run_nodes both \
      $PWD/data $PWD/results

Results
-------

The above command will save results as:

.. code-block::

    results
    ├── bycycle
    │   ├── report_group.html
    │   ├── signal_dim1-0000
    │   │   ├── report.html
    │   │   └── results.csv
    │   ├── signal_dim1-0001
    │   │   ├── report.html
    │   │   └── results.csv
    │   ├── signal_dim1-0002
    │   │   ├── report.html
    │   │   └── results.csv
    │   ├── signal_dim1-0003
    │   │   ├── report.html
    │   │   └── results.csv
    │   └── signal_dim1-0004
    │       ├── report.html
    │       └── results.csv
    └── fooof
        ├── report_group.html
        ├── spectrum_dim1-0000
        │   ├── report.html
        │   └── results.json
        ├── spectrum_dim1-0001
        │   ├── report.html
        │   └── results.json
        ├── spectrum_dim1-0002
        │   ├── report.html
        │   └── results.json
        ├── spectrum_dim1-0003
        │   ├── report.html
        │   └── results.json
        └── spectrum_dim1-0004
            ├── report.html
            └── results.json

Example html reports:

- FOOOF

  - `Individual <https://ndspflow-tools.github.io/ndspflow/results/fooof/spectrum_dim1-0000/report.html>`_
  - `Group <https://ndspflow-tools.github.io/ndspflow/results/fooof/report_group.html>`_

- bycycle

  - `Individual <https://ndspflow-tools.github.io/ndspflow/results/bycycle/signal_dim1-0000/report.html>`_
  - `Group <https://ndspflow-tools.github.io/ndspflow/results/bycycle/report_group.html>`_
