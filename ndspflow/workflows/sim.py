"""Simulations."""

import operator as op

from .utils import parse_args
from .param import check_is_parameterized

class Simulate:
    """Simulation class.

    Attributes
    ----------
    n_seconds : float
        Simulation duration, in seconds.
    fs : float
        Sampling rate, in Hertz.
    seeds : list of float, optional, default: None
        Random seeds to set. Nodes are ran per seed.
    nodes : list of list
        Contains order of operations as:
        [['simulate', function, axis, *args, **kwargs], ...]
    y_array : ndarray
        Voltage time series.
    """
    def __init__(self, n_seconds=None, fs=None, seeds=None):
        """Initalize object."""

        self.n_seconds = n_seconds
        self.fs = fs
        self.seeds = seeds
        self.y_array = None
        self.nodes = []
        self.param_ind = 0


    def simulate(self, func, *args, operator='add', **kwargs):
        """Queue simulation.

        Parameters
        ----------
        func : function
            Simulation function.
        operator : {'add', 'mul', 'sub', 'div'} or {'+', '*', '-', '/'}
            Operator to combine signals.
        *args
            Additonal positional arguments to func.
        **kwargs
            Addional keyword arguments to func.
        """

        # Evaluate funcs/gens prior to entering mp pool
        if self.seeds is not None and not isinstance(self.seeds, int):
            args, kwargs = parse_args(list(args), kwargs, self)

        is_parameterized = check_is_parameterized(args, kwargs)

        self.nodes.append(['simulate', func, args,
                           {'operator': operator}, kwargs, is_parameterized])


    def run_simulate(self, func, *args, operator='add', **kwargs):
        """Queue simulation.

        Parameters
        ----------
        func : function
            Simulation function.
        operator : {'add', 'mul', 'sub', 'div'} or {'+', '*', '-', '/'}
            Operator to combine signals.
        *args
            Additonal positional arguments to func.
        **kwargs
            Addional keyword arguments to func.
        """

        # How to combine signals
        if operator in ['add', '+']:
            operator = op.add
        elif operator in ['mul', '*']:
            operator = op.mul
        elif operator in ['sub', '-']:
            operator = op.sub
        else:
            operator = op.truediv

        # Special args/kwargs parsing
        args, kwargs = parse_args(list(args), kwargs, self)

        # Simulate
        if self.y_array is None:
            self.y_array = func(*args, **kwargs)
        else:
            self.y_array = operator(
                self.y_array, func( *args, **kwargs)
            )
