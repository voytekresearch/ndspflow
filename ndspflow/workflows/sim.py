"""Simulations."""

import types
import operator as op


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


    def simulate(self, func, *args, operator='add', **kwargs):
        """Add a simulation node.

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

            args = list(args)
            for ind in range(len(args)):
                # Get arg value if function or generator
                if isinstance(args[ind], (types.FunctionType)):
                    args[ind] = ('_iter', [args[ind]() for seed in self.seeds])
                elif isinstance(args[ind], (types.GeneratorType)) or hasattr(args[ind], '__next__'):
                    args[ind] = ('_iter', [next(args[ind])for seed in self.seeds])


            for k, v in kwargs.items():
                # Get kwarg value if function or generator
                if isinstance(v, types.FunctionType):
                    kwargs[k] = ('_iter', [v() for _ in self.seeds])
                elif isinstance(v, types.GeneratorType) or hasattr(v, '__next__'):
                    args[ind] = ('_iter', [next(v) for _ in self.seeds])

        self.nodes.append(['simulate', func, args,
                           {'operator': operator}, kwargs])


    def run_simulate(self, func, *args, operator='add', **kwargs):
        """Simulate aperiodic signal.

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
        args = list(args)

        for k, v in kwargs.items():
            if isinstance(v, str) and 'self' in v:
                kwargs[k] = getattr(self, v.split('.')[-1])
            if isinstance(v, tuple) and v[0] == '_iter':
                kwargs[k] = v[self._pind]

        for ind in range(len(args)):
            if isinstance(args[ind], str) and 'self' in args[ind]:
                args[ind] = getattr(self, args[ind].split('.')[-1])
            if isinstance(args[ind], tuple) and args[ind][0] == '_iter':
                args[ind] = args[ind][1][self._pind]


        # Simulate
        if self.y_array is None:
            self.y_array = func(*args, **kwargs)
        else:
            self.y_array = operator(
                self.y_array, func( *args, **kwargs)
            )

