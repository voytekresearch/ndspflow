"""Workflow utilities."""

import types
from copy import copy
import numpy as np


def parse_args(args, kwargs, self=None):
    """Parse args and kwargs.

    Parameters
    ----------
    args : list
        Positional parameters.
    kwargs : dict
        Keyword arguments.
    self : class, optional, default: None
        Extract argument from attributes. Parsed as 'self.attr'.

    Returns
    -------
    args : list
        Updated [ositional parameters.
    kwargs : dict
        Updated keyword arguments.
    """
    for ind in range(len(args)):
        if isinstance(args[ind], str) and 'self' in args[ind]:
            args[ind] = getattr(self, args[ind].split('.')[-1])
        elif isinstance(args[ind], (types.FunctionType)):
            args[ind] = ('_iter', [args[ind]() for _ in self.seeds])
        elif isinstance(args[ind], (types.GeneratorType)) or hasattr(args[ind], '__next__'):
            args[ind] = ('_iter', [next(args[ind]) for _ in self.seeds])
        elif isinstance(args[ind], tuple) and args[ind][0] == '_iter':
            args[ind] = args[ind][1][self.param_ind]

    for k, v in kwargs.items():
        if isinstance(v, str) and 'self' in v:
            kwargs[k] = getattr(self, v.split('.')[-1])
        elif isinstance(v, types.FunctionType):
            kwargs[k] = ('_iter', [v() for _ in self.seeds])
        elif isinstance(v, types.GeneratorType) or hasattr(v, '__next__'):
            kwargs[k] = ('_iter', [next(v) for _ in self.seeds])
        elif isinstance(v, tuple) and v[0] == '_iter':
            kwargs[k] = v[1][self.param_ind]

    return args, kwargs


def reshape(y_array, axis):
    """Numpy axis-like reshape to 2D.

    Parameters
    ----------
    y_array : ndarray
        Array to reshape.
    axis : int or tuple of int
        Axis to take 1d slices along.

    Returns
    -------
    y_array : ndarray
        Reshaped array.
    shape : tuple
        Original shape of y_array.
    """

    # Invert axis indices
    axis = [axis] if isinstance(axis, int) else axis
    axes = list(range(len(y_array.shape)))
    axis = [axes[ax] for ax in axis]
    axis = tuple([ax for ax in axes if ax not in axis])

    # Track original shape to later reshape results
    shape = [s for i, s in enumerate(y_array.shape) if i in axis]

    # Reshape to 2d based on axis argument
    #   this allows passing slices to mp pools
    n_axes = len(axis)
    y_array = np.moveaxis(y_array, axis, list(range(n_axes)))
    y_array = y_array.reshape(*[-1, *y_array.shape[n_axes:]])

    return y_array, shape