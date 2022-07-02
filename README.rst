========
ndspflow
========

|CircleCI|_ |Codecov|_

.. |CircleCI| image:: https://circleci.com/gh/voytekresearch/ndspflow.svg?style=svg&circle-token=b26555544cf83f79a4aa45f6f4b98423e2ee06d0
.. _CircleCI: https://circleci.com/gh/voytekresearch/ndspflow

.. |Codecov| image:: https://codecov.io/gh/voytekresearch/ndspflow/branch/master/graph/badge.svg?token=I9Z7OPIZ7J
.. _Codecov: https://codecov.io/gh/voytekresearch/ndspflow

A workflow manager for processing and modeling neural timeseries

Dependencies
------------

- numpy >= 1.22.4
- fooof >= 1.0.0
- bycycle @ git+https://github.com/bycycle-tools/bycycle.git@main
- plotly >= 4.10.0
- scikit-learn == 0.24.1
- scikit-image == 0.18.1
- emd == 0.4.0

Motivations
-----------

Ndspflow provides a framework to define and execute neural data analysis workflows, from raw data to final model.

- Reproducible and standardized analyses
- Compatible with BIDS, simulated, or user-defined input signals
- Interfaces with any DSP package (scipy, numpy, mne, neurodsp, etc.)
- Supports a variety of models (fooof, bycycle, scikit-learn, etc.)
- Full workflow parallelization.

Installation
------------

.. code-block:: shell

    $ git clone git@github.com:voytekresearch/ndspflow.git
    $ cd ndspflow
    $ pip install .

Quickstart
----------




Funding
-------

Supported by NIH award R01 GM134363

`NIGMS <https://www.nigms.nih.gov/>`_

.. image:: https://www.nih.gov/sites/all/themes/nih/images/nih-logo-color.png
  :width: 400

|
