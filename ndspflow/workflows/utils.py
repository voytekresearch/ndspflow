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
