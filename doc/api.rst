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

Workflow
~~~~~~~~

Core object for defining and executing analysis workflows.
This class sub-classes :class:`~workflow.BIDS`, :class:`~workflow.Simulation`,
:class:`~workflow.Transform`, and :class:`~workflow.Model` classes.

.. currentmodule:: ndspflow.workflows

.. autosummary::
    :toctree: generated/

    WorkFlow

BIDS
~~~~

An interface for reading BIDS organized signals into a :class:`~workflow.WorkFlow`.

.. autosummary::
    :toctree: generated/

    BIDS

Simulation
~~~~~~~~~~

An interface allowing simulated signal inputs into a :class:`~workflow.WorkFlow`.

.. autosummary::
    :toctree: generated/

    Simulate


Transformations
~~~~~~~~~~~~~~~

An interface for defining any (series of) array transform of the input data within a :class:`~workflow.WorkFlow`.

.. autosummary::
    :toctree: generated/

    Transform

Models
~~~~~~

An wrapper for any model class with a fit method.

.. autosummary::
    :toctree: generated/

    Model


Graphs
~~~~~~

Functions to generate workflow graphs using networkx.

.. autosummary::
    :toctree: generated/

    create_graph
    inspect_workflow


Models
~~~~~~

Fits FOOOF and Bycycle models for 1D, 2D or 3D arrays.

.. autosummary::
    :toctree: generated/

    fit_fooof
    fit_bycycle
