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
    def __init__(self, y_arr=None, x_arr=None, **kwargs):
        """Initalize object.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments that sub-classes
            need access to.
        """

        # Initialize sub-classes
        Simulate.__init__(self)
        Model.__init__(self)

        # Initialize self
        self.nodes = []
        self.y_arr = y_arr
        self.x_arr = x_arr
        self.return_attrs = None

        # Set sub-class attributes
        for k, v in kwargs.items():
            setattr(self, k, v)


    def run(self, return_attrs=None, n_jobs=-1, progress=None):
        """Run workflow."""
        
        self.return_attrs = return_attrs

        if self.seeds is None:
            for node in self.nodes:
                if node[0] in ['simulate', 'transform']:
                    getattr(self, 'run_' + node[0])(node[1], *node[2],
                                                    **node[3], **node[4])
                elif node[0] == 'fit':
                    getattr(self, 'run_' + node[0])(self.x_arr, self.y_arr,
                                                    *node[2], **node[3])
                else:
                    getattr(self, 'run_' + node[0])(*node[1], **node[2])
        else:
            n_jobs = cpu_count() if n_jobs == -1 else n_jobs
            with Pool(processes=n_jobs) as pool:
                mapping = pool.imap(self._run, self.seeds)

                if progress is not None:
                    results = list(progress(mapping, total=len(self.seeds),
                                            desc='Running Workflow'))
                else:
                    results = list(mapping)
            self.results = results


    def _run(self, seed):
        """Sub-function to allow imap parallelziation.

        Parameters
        ----------
        seed : int
            Random seed to set.
        """
        np.random.seed(seed)

        # Clear
        for node in self.nodes:
            if node[0] in ['simulate', 'transform']:
                getattr(self, 'run_' + node[0])(node[1], *node[2],
                                                **node[3], **node[4])
            elif node[0] == 'fit':
                getattr(self, 'run_' + node[0])(self.x_arr, self.y_arr,
                                                *node[2], **node[3])
            else:
                getattr(self, 'run_' + node[0])(*node[1], **node[2])

        if isinstance(self.return_attrs, str):
            return getattr(self, self.return_attrs)
        elif self.return_attrs is None:
            return None
        else:
            return [getattr(self, i) if i not in ['self', 'model_self'] 
                    else getattr(self, 'model_self') for i in self.return_attrs]
