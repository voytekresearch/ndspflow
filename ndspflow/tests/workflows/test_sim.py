"""Test simulation class."""

import numpy as np
import pytest
from neurodsp.sim import sim_oscillation
from ndspflow.workflows import Simulate


def test_Simulate_init():

    # Settings
    n_seconds = 10
    fs = 1000
    seeds = np.arange(10)

    # Test initialization
    sim = Simulate(n_seconds, fs, seeds)

    assert sim.n_seconds == n_seconds
    assert sim.fs == fs
    assert (seeds == sim.seeds).all()
    assert sim.y_array is None
    assert len(sim.nodes) == 0


def test_Simulate_simulate():

    # Settings
    n_seconds = 10
    fs = 1000
    seeds = np.arange(10)

    # Test queue
    np.random.seed(0)

    sim = Simulate(n_seconds, fs, seeds)

    # Iterable args
    def freq_func(): return np.random.choice([10, 20])
    freq_gen = (np.random.choice([10, 20]) for _ in seeds)

    # Iterable kwargs
    def phase_func(): return np.random.choice(['min', 'max'])
    phase_gen = (np.random.choice(['min', 'max']) for _ in seeds)

    sim.simulate(sim_oscillation, 'self.n_seconds', 'self.fs', freq_func, phase=phase_func)
    sim.simulate(sim_oscillation, 'self.n_seconds', 'self.fs', freq_gen, phase=phase_gen)

    assert len(sim.nodes) == 2
    assert sim.nodes[0][0] == 'simulate'
    assert callable(sim.nodes[0][1])
    assert sim.nodes[0][2][0] == sim.n_seconds
    assert sim.nodes[0][2][1] == sim.fs
    assert sim.nodes[0][2][2][0] == '_iter'
    assert len(sim.nodes[0][2][2][1]) == len(seeds)
    assert isinstance(sim.nodes[0][3], dict)
    assert isinstance(sim.nodes[0][4], dict)


@pytest.mark.parametrize('operator', ['+', '-', '*', '/'])
def test_Simulate_run_simulate(operator):

    # Settings
    n_seconds = 10
    fs = 1000
    seeds = np.arange(10)

    # Test run
    np.random.seed(0)

    sim = Simulate(n_seconds, fs, seeds)

    sim.phase = 'max'

    sim.run_simulate(sim_oscillation, 'self.n_seconds', 'self.fs', ('_iter', [10, 20]),
                     phase='self.phase')
    sim.run_simulate(sim_oscillation, 'self.n_seconds', 'self.fs', ('_iter', [10, 20]),
                     operator=operator, phase=('_iter', ['min', 'max']))

    assert isinstance(sim.y_array, np.ndarray)
    assert len(sim.y_array) == int(n_seconds * fs)
