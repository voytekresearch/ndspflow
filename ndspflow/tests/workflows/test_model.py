"""Test model wrapper."""

from ndspflow.workflows import  Model
from ndspflow.workflows.model import Merge, Result
from fooof import FOOOF
from bycycle import Bycycle


def test_Model(test_data):

    powers = test_data['powers_3d']
    freqs = test_data['freqs']
    fs = test_data['fs']

    # Queue
    model = Model()
    model.fit(FOOOF(max_n_peaks=1), freqs, powers, (1, 100), axis=-1)
    assert len(model.nodes) == 1

    # Execute
    model.run_fit(freqs, powers, (1, 100), axis=-1)

    model.node = model.nodes[0]
    model.run_fit(freqs, powers, (1, 100), axis=-1)
    assert isinstance(model.models[0].result[0], FOOOF)

    # 1d
    powers = test_data['powers_1d']
    model = Model()
    model.fit(FOOOF(max_n_peaks=1), freqs, powers, (1, 100))
    model.run_fit(freqs, powers, (1, 100))
    assert isinstance(model.models[0].result, FOOOF)

    # No x-array
    sig = test_data['sig_3d']
    model = Model()
    model.fit(Bycycle(), sig, fs, (1, 100), axis=-1)
    model.run_fit(None, sig, fs, (5, 50), axis=-1)
    assert isinstance(model.models[0].result[0], Bycycle)

    # 1d
    sig = test_data['sig_1d']
    model = Model()
    model.fit(Bycycle(), sig, fs, (1, 100))
    model.run_fit(None, sig, fs, (5, 50))
    assert isinstance(model.models[0].result, Bycycle)

def test_Merge():

    merge = Merge()
    merge.fit([0, 1, 2], [0, 1, 2])
    assert merge._x_array is not None
    assert merge._y_array is not None

    merge = Merge()
    merge.fit([0, 1, 2])
    assert merge._x_array is None
    assert merge._y_array is not None

def test_Result():

    res = Result(0)
    assert res.result == 0
