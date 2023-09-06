"""Workflow utilities."""

import os
import re
import inspect

import types
from inspect import signature
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
        elif self.seeds is not None and \
            (isinstance(args[ind], (types.GeneratorType)) or hasattr(args[ind], '__next__')):
            args[ind] = ('_iter', [next(args[ind]) for _ in self.seeds])
        elif isinstance(args[ind], tuple) and args[ind][0] == '_iter':
            args[ind] = args[ind][1][self.param_ind]

    for k, v in kwargs.items():
        if isinstance(v, str) and 'self' in v:
            kwargs[k] = getattr(self, v.split('.')[-1])
        elif self.seeds is not None and isinstance(v, types.FunctionType):
            kwargs[k] = ('_iter', [v() for _ in self.seeds])
        elif self.seeds is not None and isinstance(v, types.GeneratorType) or hasattr(v, '__next__'):
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


def extract_results(models, attrs=None, flatten=False):
    """Extract results from model attributes.

    Parameters
    ----------
    models : list of model.Result
        Contains fit model.
    attrs : list of str or str
        Defines attribute(s) to extract from models.

    Notes
    -----
    The attrs implicitly extracts attributes based on its shape and the shape of models.

    attrs = ['some_attr'] and attrs = 'some_attr' :
        Extracts the same attribute per model.
    attrs = ['attr_0', 'attr_1'] :
        Each attribute is extracted from each model.
    attrs = [['attr_0_model0', 'attr_0_model0'], ['attr_0_model1', 'attr_0_model1']]
        Extracts unique attributes (2) per unique model (2).

    """

    unpack_dict = False

    if isinstance(attrs, str):
        # Single and same attribute extracted from all models
        results = [getattr(model.result, attrs) for model in models]
    elif isinstance(attrs, list) and isinstance(attrs[0], str):
        # 1d attributes, same for each model
        results = [[getattr(model.result, r) for r in attrs] for model in models]
    elif isinstance(attrs, list) and isinstance(attrs[0], list):
        # 2d attributes, unique for each model
        results=  [{r: getattr(model.result, r) for r in attrs[i]}
                   for i, model in enumerate(models)]
        unpack_dict = True
    elif attrs is None and models is not None:
        return models
    else:
        return None

    # Unpack
    if len(models) == 1:
        results = results[0]

    # Flatten into 2d array
    if len(models) == 1 and flatten and not unpack_dict:
        results = np.hstack([j.flatten() for j in results])
    elif len(models) == 1 and flatten:
        results = np.array([np.hstack([np.array([*j.values()]).flatten() for j in results])])
    elif flatten and not unpack_dict:
        results = np.array([np.hstack([j.flatten() for j in i]) for i in results])
    elif flatten:
        results = np.hstack([i[j].flatten() for i in results for j in i])

    return results


def get_init_params(model):
    """Get model initialization parameters.

    Parameters
    ----------
    model : class
        Model object with a .fit method.

    Returns
    -------
    params_init : list of str
        Names of parameters that are accepts by the model's init or
        by the model's super class.
    """

    # Search current class
    params_init = [i for i in list(signature(model.__init__).parameters)
                   if i not in ['self', 'args', 'kwargs']]

    # Search super class(es)
    params_init.extend(
        [j for i in model.__class__.__bases__ for j in
        list(signature(i.__init__).parameters) if j not in ['self', 'args', 'kwargs']]
    )

    return params_init


def import_function(func):
    """Imports function defined on main. Required for multiprocessing with
    functions defined in jupyter.

    Parameters
    ----------
    func : function
        Function defined on __main__.

    Returns
    -------
    func : function
        Function defined from temporary import.
    """

    # Move func to a .py file
    if os.path.isfile('_tmp_funcs_mp.py'):
        os.remove('_tmp_funcs_mp.py')

    lines = inspect.getsource(func)

    # Ensure consistent func name for importing
    func_name = func.__name__

    lines = re.sub(f'def {func_name}', 'def func', lines)

    # Write function
    with open('_tmp_funcs_mp.py', 'w') as f:
        for line in lines:
            f.write(line)

    # Import
    from _tmp_funcs_mp import func

    return func
