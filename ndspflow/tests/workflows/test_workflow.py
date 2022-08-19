"""Workflow tests."""

import pytest
import numpy as np

from ndspflow.tests.utils import pbar
from neurodsp.sim import sim_oscillation, sim_powerlaw
from neurodsp.spectral import compute_spectrum
from fooof import FOOOF
from ndspflow.workflows import WorkFlow


# test transform functions
def _t(y): return 2*y
def _z(y): return 0*y
def _m(y): return y.mean()

def test_workflow_transform(bids_path):

    # test np.array input: 1d
    y_array = np.random.rand(10000)
    wf = WorkFlow(y_array=y_array.copy())
    wf.transform(_t)
    wf.run()
    assert (wf.y_array == (y_array * 2)).all()

    # test np.array input: 2d
    y_array = np.random.rand(2, 5000)
    wf = WorkFlow(y_array=y_array.copy())
    wf.transform(_t)
    wf.run(axis=-1, progress=pbar)
    assert (wf.y_array == (y_array * 2)).all()

    y_array = np.random.rand(2, 5000)
    wf = WorkFlow(y_array=y_array.copy())
    wf.transform(_t)
    wf.run(axis=None)
    assert (wf.y_array == (y_array * 2)).all()

    # test np.array input: 3d
    y_array = np.random.rand(2, 3, 5000)
    wf = WorkFlow(y_array=y_array.copy())
    wf.transform(_m, axis=-1)
    wf.run()
    assert (wf.y_array == y_array.mean(axis=-1)).all()

    # raise error: no input
    with pytest.raises(ValueError):
        wf =  WorkFlow()
        wf.run()


def test_workflow_simulate():

    # test simulation
    wf = WorkFlow(seeds=np.arange(3))
    wf.simulate(sim_oscillation, 10, 1000, 10)
    wf.run()
    assert wf.y_array.shape == (3, 10000)

    # without


def test_workflow_bids(bids_path):
    # test BIDS: queue
    wf =  WorkFlow(bids_path=bids_path, session='01',
                   datatype='ieeg', task='test', extension='.vhdr')
    wf.read_bids(subject='01', queue=True, allow_ragged=True)
    wf.transform(_z)
    wf.run()
    assert wf.y_array.sum() == 0

    # test BIDS: no queue
    wf =  WorkFlow(bids_path=bids_path, session='01',
                   datatype='ieeg', task='test', extension='.vhdr')
    wf.read_bids(subject='01', queue=False, allow_ragged=True)
    wf.transform(_z)
    wf.run()
    assert wf.y_array.sum() == 0


def test_workflow_transform_fit():

    # test models: return model
    wf = WorkFlow(seeds=np.arange(3))
    wf.simulate(sim_oscillation, 10, 1000, 10)
    wf.simulate(sim_powerlaw, 10, 1000)
    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=1, verbose=False))
    wf.run()
    assert isinstance(wf.results[0], FOOOF)

    # test models: return attribute
    wf = WorkFlow(seeds=np.arange(3))
    wf.simulate(sim_oscillation, 10, 1000, 10)
    wf.simulate(sim_powerlaw, 10, 1000)
    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=1, verbose=False))
    wf.run(attrs='peak_params_')
    assert isinstance(wf.results[0], np.ndarray)

    # 3d with fooof
    y_array = np.random.rand(2, 3, 5000)
    wf = WorkFlow(y_array=y_array.copy())
    wf.transform(_t)
    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=1, verbose=False))
    wf.run(axis=-1)

    assert isinstance(wf.results[0][0], FOOOF)
    assert wf.results[0][0].has_data and wf.results[0][0].has_model
    assert wf.results.shape == y_array.shape[:-1]

def test_workflow_fork():


    # test forks
    wf = WorkFlow(seeds=np.arange(3))
    wf.simulate(sim_oscillation, 10, 1000, 10)
    wf.simulate(sim_powerlaw, 10, 1000)
    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=0))
    wf.fork(0)
    wf.fit(FOOOF(max_n_peaks=1))

    wf.run()

    # test .run
    # test ._run
    # test .fork
    # test .run_fork
    # test .fit_transform
    # test plot