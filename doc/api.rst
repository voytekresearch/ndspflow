.. _api_documentation:

=================
API Documentation
=================

API reference for the ndspflow module.

Table of Contents
=================

.. contents::
    :local:
    :depth: 2

.. currentmodule::ndspflow

Workflows
~~~~~~~~~

Create a nipype workflow connecting input, FOOOF and Bycycle nodes.

.. currentmodule:: ndspflow.core.workflows

.. autosummary::
    :toctree: generated/

    create_workflow
    wf_fooof
    wf_bycycle

Interfaces
~~~~~~~~~~

Input, output, and runtime definitions for the FOOOF and Bycycle nodes.

.. currentmodule:: ndspflow.core.interfaces

.. autosummary::
    :toctree: generated/

    FOOOFNodeInputSpec
    FOOOFNodeOutputSpec
    FOOOFNode
    BycycleNodeInputSpec
    BycycleNodeOutputSpec
    BycycleNode

Model Fitting
~~~~~~~~~~~~~

Fits FOOOF and Bycycle models for 1D, 2D or 3D arrays.

.. currentmodule:: ndspflow.core.fit

.. autosummary::
    :toctree: generated/

    fit_fooof
    fit_bycycle

Plotting
~~~~~~~~

Create FOOOF and Bycycle plots that will be embedded in html reports.

.. currentmodule:: ndspflow.plts.fooof

.. autosummary::
    :toctree: generated/

    plot_fm
    plot_fg
    plot_fgs

.. currentmodule:: ndspflow.plts.bycycle

.. autosummary::
    :toctree: generated/

    plot_bm
    plot_bg
    plot_bgs

Reports
~~~~~~~

Generate analysis html reports.

.. currentmodule:: ndspflow.reports.html

.. autosummary::
    :toctree: generated/

    generate_report
    generate_header
    generate_fooof_report
    generate_bycycle_report


Input/Output
~~~~~~~~~~~~

Check input and output directories and model saving.

.. currentmodule:: ndspflow.io.paths

.. autosummary::
    :toctree: generated/

    check_dirs
    clean_mkdir

.. currentmodule:: ndspflow.io.save

.. autosummary::
    :toctree: generated/

    save_fooof
    save_bycycle

Command-Line Interface
~~~~~~~~~~~~~~~~~~~~~~

Parses analysis arguments/settings from command-line calls.

.. currentmodule:: ndspflow.cli.ndspflow_run

.. autosummary::
    :toctree: generated/

    get_parser
