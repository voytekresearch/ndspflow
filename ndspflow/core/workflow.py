"""Workflows."""

from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

from .sim import Simulate
from .transform import Transform
from .model import Model


class WorkFlow(Simulate, Transform, Model):
    """Workflow definition.

    Attributes
    ----------
    y_arr : ndarray
        Y-axis values. Usually voltage or power.
    x_arr : 1d array, optional, default: None
        X-axis values. Usually time or frequency.
    nodes : list of list
        Contains order of operations as:
        [[node_type, function, *args, **kwargs], ...]
    """
    def __init__(self):
        """Initalize object."""
        self.nodes = []
        self.y_arr = None
        self.x_arr = None

        self.seeds = None


    def simulate_init(self, n_seconds, fs, seeds=None):
        """Initalize simulation attributes."""
        self.n_seconds = n_seconds
        self.fs = fs
        self.seeds = seeds


    def fit_init(self, model, *args, **kwargs):
        """Initialze model to fit."""
        self.model = Model(model, *args, **kwargs)


    def run(self, return_attrs=None, n_jobs=-1, progress=None):
        """Run workflow."""

        if self.seeds is None:
            for node in self.nodes:
                if node[0] != 'fit':
                    getattr(self, 'run_' + node[0])(node[1], node[2], *node[3], **node[4])
                else:
                    getattr(self, 'run_' + node[0])(self.x_arr, self.y_arr, *node[1], **node[2])
        else:
            n_jobs = cpu_count() if n_jobs == -1 else n_jobs
            with Pool(processes=n_jobs) as pool:
                mapping = pool.imap(partial(self._run, return_attrs=return_attrs),
                                    self.seeds)

                if progress is not None:
                    results = list(progress(mapping, total=len(self.seeds),
                                            desc='Running Workflow'))
                else:
                    results = list(mapping)
            self.results = results


    def _run(self, seed, return_attrs=None):
        """Sub-function to allow imap parallelziation.

        Parameters
        ----------
        seed : int
            Random seed to set.
        """
        np.random.seed(seed)

        # Clear
        for node in self.nodes:
            if node[0] != 'fit':
                getattr(self, 'run_' + node[0])(node[1], node[2], *node[3], **node[4])
            else:
                getattr(self, 'run_' + node[0])(self.x_arr, self.y_arr, *node[1], **node[2])

        if isinstance(return_attrs, str):
            return getattr(self, return_attrs)
        elif return_attrs is None:
            return None
        else:
            return [getattr(self, i) for i in return_attrs]
