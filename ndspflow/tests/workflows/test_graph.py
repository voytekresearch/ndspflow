"""Test workflow graphs."""

import pytest
import numpy as np

from neurodsp.filt import filter_signal
from neurodsp.sim import sim_oscillation
from neurodsp.spectral import compute_spectrum

from fooof import FOOOF

from ndspflow.workflows import WorkFlow
from ndspflow.workflows.graph import create_graph, inspect_workflow


@pytest.mark.parametrize('forks', [True, False])
def test_create_graph(bids_path, forks):

    # BIDS input
    subjects, fs = ['01'], 1000

    wf = WorkFlow(bids_path=bids_path, subjects=subjects, fs=fs, session='01',
                  datatype='ieeg', task='test', extension='.vhdr')

    wf.read_bids(allow_ragged=True)

    wf.transform(filter_signal, fs, 'lowpass', 200, remove_edges=False)

    if forks:
        wf.fork(0)
        wf.fit(FOOOF())
        wf.fork(0)

    wf.fit(FOOOF())

    graph = create_graph(wf)

    if not forks:
        assert list(graph.nodes) == ['read', 'trans00', 'fit00']
        assert list(graph.edges) == [('read', 'trans00'), ('trans00', 'fit00')]

    # Simulation input
    wf = WorkFlow(seeds=np.arange(3))

    wf.simulate(sim_oscillation, 10, 1000, 10)

    if forks:
        wf.fork(0)
        wf.transform(compute_spectrum, 1000)
        wf.fork(0)
        wf.transform(compute_spectrum, 1000)

        wf.fork(1)
        wf.fit(FOOOF(max_n_peaks=1))
        wf.fork(1)

    wf.fit(FOOOF(max_n_peaks=2))

    graph = create_graph(wf)

    if forks:
        assert list(graph.nodes) == ['sim', 'trans00', 'trans01', 'fit00', 'fit01']
        assert list(graph.edges) == [('sim', 'trans00'), ('sim', 'trans01'),
                                     ('trans01', 'fit00'), ('trans01', 'fit01')]
