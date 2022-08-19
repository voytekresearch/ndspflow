"""Test transformations."""

import pytest
import numpy as np
from ndspflow.workflows import Transform
from ndspflow.workflows.transform import func_wrapper


@pytest.mark.parametrize('x_array', [True, False])
def test_Transform(x_array):

    y_array = np.random.rand(1000)

    if x_array:
        x_array = np.arange(len(y_array))
        def tfunc(x, y): return y**2
    else:
        x_array = None
        def tfunc(y): return y**2

    # Init
    trans = Transform(y_array, x_array)
    assert len(trans.nodes) == 0
    assert (trans.y_array == y_array).all()

    # Queue
    trans = trans.transform(tfunc)

    # Run
    trans = Transform(y_array, x_array)
    trans.run_transform(tfunc)
    assert (trans.y_array == y_array**2).all()


@pytest.mark.parametrize('axis', [0, 1, 2, (0, 1), (0, 2), (1, 2), -1])
def test_Transform_multidim(axis):

    # Multi-dim
    y_array = np.random.rand(2, 3, 1000)
    x_array = None
    def tfunc(y): return np.mean(y)

    trans = Transform(y_array, x_array)
    trans.run_transform(tfunc, axis=axis)
    print(trans.y_array.shape, np.mean(y_array, axis=axis).shape)
    assert (trans.y_array == np.mean(y_array, axis=axis)).all()

    # Test in place
    trans = Transform(y_array, x_array)
    def tfunc(y): return y + 1
    trans.run_transform(tfunc, axis=axis)


@pytest.mark.parametrize('x', [True, False])
@pytest.mark.parametrize('return_x', [True, False])
def test_func_wrapper(x, return_x):

    y = np.random.rand(2, 1000)

    if x:
        x = np.arange(len(y))
    else:
        x = None

    if x is not None and return_x:
        def tfunc(x, y): return x, y
    elif x is not None:
        def tfunc(x, y): return y
    else:
        def tfunc(y): return y

    x, y = func_wrapper(tfunc, x, y)
