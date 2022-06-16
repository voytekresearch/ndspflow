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
        self.models = []
        
        # Initialize sub-classes
        Simulate.__init__(self)
        Model.__init__(self)

        # Initialize self
        self.nodes = []
        self.models = []

        self.node = None
        self.mdoel = None

        self.y_arr = y_arr
        self.x_arr = x_arr

        self.y_arr_stash = None
        self.x_arr_stash = None
        self.fork_inds = []
        
        self.results = None
        self.return_attrs = None

        # Set sub-class attributes
        for k, v in kwargs.items():
            setattr(self, k, v)


    def run(self, return_attrs=None, n_jobs=-1, progress=None):
        """Run workflow."""
     
        if self.fork_inds is not None:
            self.y_arr_stash = [None] * len(self.fork_inds)
            self.x_arr_stash = [None] * len(self.fork_inds)

        self.return_attrs = return_attrs

        if self.seeds is None:
            # FIX this for non-seeded simulations
            pass
        else:
            n_jobs = cpu_count() if n_jobs == -1 else n_jobs
            with Pool(processes=n_jobs) as pool:
                mapping = pool.imap(self._run, self.seeds)

                if progress is not None:
                    _results = list(progress(mapping, total=len(self.seeds),
                                             desc='Running Workflow'))
                else:
                    _results = list(mapping)

            if self.results is not None:
                self.results.append(_results)
            else:
                self.results = _results

        # Reset temporary attributes
        self.model = None
        self.node = None

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
            self.node = node
            if node[0] in ['simulate', 'transform']:
                getattr(self, 'run_' + node[0])(node[1], *node[2],
                                                **node[3], **node[4])
            elif node[0] == 'fit':
                getattr(self, 'run_' + node[0])(self.x_arr, self.y_arr,
                                                *node[2], **node[3])
            elif node[0] == 'fork':
                getattr(self, 'run_' + node[0])(node[1])
            else:
                getattr(self, 'run_' + node[0])(*node[1], **node[2])
        
        # if isinstance(self.return_attrs, str):
        #     return getattr(self, self.return_attrs)
        # elif self.return_attrs is None:
        #     return None
        # else:
            
        if self.models is not None:
            return self.models

            # FIX: for single models and specific attribute returns
            # return [getattr(self, i) if i not in ['self', 'model_self'] 
            #         else getattr(self, 'model_self') for i in self.return_attrs]
    
    def fork(self, ind=0):
        """Queue fork.
        
        Parameters
        ----------
        ind : int
            Reference to fork to rewind to.
        """
        if ind not in self.fork_inds:
            self.fork_inds.append(ind)

        self.nodes.append(['fork', ind])

    def run_fork(self, ind=0):
        """Execute fork.
        
        Parameters
        ----------
        ind : int
            Reference to fork to rewind to.
        """

        if self.y_arr_stash[ind] is None:
            # Stash
            self.y_arr_stash[ind] = self.y_arr.copy()
            if self.x_arr is not None:
                self.x_arr_stash[ind] = self.x_arr.copy()
        elif self.y_arr_stash[ind] is not None:
            # Pop
            self.y_arr = self.y_arr_stash[ind]
            self.x_arr = self.x_arr_stash[ind]
