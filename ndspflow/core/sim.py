"""Simulations."""

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
    y_arr : ndarray
        Voltage time series.
    """
    def __init__(self, n_seconds, fs, seeds=None):
        """Initalize object."""

        self.n_seconds = n_seconds
        self.fs = fs
        self.seeds = seeds
        self.y_arr = None
        self.nodes = []


    def simulate(self, func, operator='add', *args, **kwargs):
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
        self.nodes.append(['simulate', func, operator, args, kwargs])


    def run_simulate(self, sim_func, operator='add', *sim_args, **sim_kwargs):
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

        # Simulate
        if self.y_arr is None:
            self.y_arr = sim_func(self.n_seconds, self.fs, *sim_args, **sim_kwargs)
        else:
            self.y_arr = operator(
                self.y_arr, sim_func(self.n_seconds, self.fs, *sim_args, **sim_kwargs)
            )
