"""Workflow tests."""

import pytest
import numpy as np

from ndspflow.tests.utils import plot_test, pbar, FitPass
from neurodsp.sim import sim_oscillation, sim_powerlaw
from neurodsp.spectral import compute_spectrum

from fooof import FOOOF
from bycycle import Bycycle
from sklearn.decomposition import PCA

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

    wf.fork(0)
    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=0))

    wf.fork(0)
    wf.transform(compute_spectrum, 1000)

    wf.fork(1)
    wf.fit(FOOOF(max_n_peaks=1))

    wf.fork(1)
    wf.fit(FOOOF(max_n_peaks=2))

    wf.run()

    assert np.product(wf.results.shape) == 9


def test_workflow_merge():

    # test merge
    wf_merge = WorkFlow(seeds=np.arange(3))

    wf_merge.simulate(sim_powerlaw, 10, 1000)

    wf_merge.fork(0)
    wf_merge.simulate(sim_oscillation, 10, 1000, 10)
    wf_merge.merge()

    wf_merge.fork(0)
    wf_merge.simulate(sim_oscillation, 10, 1000, 20)
    wf_merge.merge()

    wf_merge.transform(compute_spectrum, 1000)
    wf_merge.fit(FOOOF(max_n_peaks=1, verbose=False))

    wf_merge.run(attrs='peak_params_')

    # compare to non-merged equivalent
    wf = WorkFlow(seeds=np.arange(3))

    wf.simulate(sim_powerlaw, 10, 1000)

    wf.fork(0)
    wf.simulate(sim_oscillation, 10, 1000, 10)
    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=1, verbose=False))

    wf.fork(0)
    wf.simulate(sim_oscillation, 10, 1000, 20)
    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=1, verbose=False))

    wf.run(attrs='peak_params_')

    assert (wf_merge.results == wf.results).all()


def test_workflow_results_shape():

    # test: return ragged shaped results from two models
    wf = WorkFlow(seeds=np.arange(3))

    wf.simulate(sim_powerlaw, 10, 1000)
    wf.simulate(sim_oscillation, 10, 1000, 10)

    wf.fork(0)
    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=1, verbose=False))

    wf.fork(0)
    wf.fit(Bycycle(), 1000, (1, 100))

    wf.run(attrs=[['peak_params_'], ['monotonicity']])

    assert wf.results.dtype == 'object'
    assert list(wf.results[0][0].keys())[0] == 'peak_params_'
    assert list(wf.results[0][1].keys())[0] == 'monotonicity'

    # test: return ragged shaped results from one model
    wf = WorkFlow(seeds=np.arange(3))

    wf.simulate(sim_powerlaw, 10, 1000)

    wf.simulate(sim_oscillation, 10, 1000, 10)
    wf.simulate(sim_oscillation, 10, 1000, 20)

    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=2, verbose=False))

    wf.run(attrs=['peak_params_', 'aperiodic_params_'])

    assert len(wf.results) == 3 # 3 sims
    assert len(wf.results[0]) == 2 # 2 returned attributes
    assert wf.results[0][0].shape == (2, 3) # 2 peaks x 3 gaussian params
    assert len(wf.results[0][1]) == 2 # 2 aperiodic params (offset, slope)

    # test: null return
    wf = WorkFlow(seeds=np.arange(3))

    wf.simulate(sim_powerlaw, 10, 1000)
    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=0, verbose=False))

    wf.run(attrs=[None])

    assert (wf.results == None).all()

def test_workflow_results_flatten():

    # Test flattening results
    wf = WorkFlow(seeds=np.arange(3))

    wf.simulate(sim_powerlaw, 10, 1000)
    wf.simulate(sim_oscillation, 10, 1000, 10)

    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=1, verbose=False))

    wf.run(n_jobs=1, attrs='aperiodic_params_', flatten=True)

    # 3 simulations x 2 aperiodic params
    assert wf.results.shape == (3, 2)

    # 3 simulation x 4 (a)periodic params
    wf.run(attrs=['aperiodic_params_', 'peak_params_'], flatten=True, n_jobs=1)
    assert wf.results.shape == (3, 5)


def test_workflow_fit_transform():

    # Transform y_array to (a)periodic params and pass to new model
    wf = WorkFlow(seeds=np.arange(3))

    wf.simulate(sim_powerlaw, 10, 1000)
    wf.simulate(sim_oscillation, 10, 1000, 10)

    wf.transform(compute_spectrum, 1000)

    wf.fit_transform(FOOOF(max_n_peaks=1, verbose=False),
                     y_attrs=['peak_params_', 'aperiodic_params_'], queue=True)

    wf.fit(FitPass())

    wf.run(attrs='params')

    assert wf.results.shape == (3, 5)

    # Pass FOOOF params (as y_array) to the fit_transform method of PCA
    wf = WorkFlow(seeds=np.arange(3))

    wf.simulate(sim_powerlaw, 10, 1000)
    wf.simulate(sim_oscillation, 10, 1000, 10)

    wf.transform(compute_spectrum, 1000)

    wf.fit_transform(FOOOF(max_n_peaks=1, verbose=False),
                    y_attrs=['peak_params_', 'aperiodic_params_'])

    wf.fit_transform(PCA(n_components=2))

    assert wf.y_array.shape == (3, 2)


def test_workflow_drop():

    wf = WorkFlow()

    wf.x_array = [0, 1]
    wf.y_array = [0, 1]

    wf.drop_y()
    wf.drop_x()

    assert wf.x_array is None and wf.y_array is None

@plot_test
def test_workflow_plot():

    wf = WorkFlow(seeds=np.arange(3))

    wf.simulate(sim_powerlaw, 10, 1000)
    wf.simulate(sim_oscillation, 10, 1000, 10)

    wf.fork(0)
    wf.transform(compute_spectrum, 1000)
    wf.fit(FOOOF(max_n_peaks=1, verbose=False))

    wf.fork(0)
    wf.fit(Bycycle(), 1000, (1, 100))

    wf.plot()
