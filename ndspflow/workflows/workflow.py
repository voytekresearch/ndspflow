"""Workflows."""

from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from .sim import Simulate
from .transform import Transform
from .graph import create_graph


class WorkFlow(Simulate, Transform):
    """Workflow definition.

    Attributes
    ----------
    models : list
        Fit model objects.
    results : list, optional
        Attributes from model classes.
    graph : networkx.DiGraph
        Directed workflow graph.
    nodes : list of list
        Contains order of operations as:
        [[node_type, function, *args, **kwargs], ...]
    """
    def __init__(self, y_arr=None, x_arr=None, **kwargs):
        """Initalize object.

        Parameters
        ----------
        y_arr : ndarray, optional, default: None
            Y-axis values. Usually voltage or power.
        x_arr : 1d array, optional, default: None
            X-axis values. Usually time or frequency.
        **kwargs
            Additional keyword arguments that sub-classes
            need access to.
        """
        # Initialize sub-classes
        Simulate.__init__(self)

        # Initialize self
        self.nodes = []
        self.models = []

        self.node = None
        self.model = None

        self.graph = None

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
        """Run workflow.

        Parameters
        ----------
        return_attrs : list of str
            Model attributes to return
        n_jobs : int, optional, default: -1
            Number of jobs to run in parallel.
        progress : {None, tqdm.notebook.tqdm, tqdm.tqdm}
            Progress bar.
        """
        # Reset pre-existing results
        if self.results is not None:
            self.results = None

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

        if isinstance(self.return_attrs, str):
            # Single and same attribute extracted from all models
            return [getattr(model,self.return_attrs) for model in self.models]
        elif isinstance(self.return_attrs, list) and isinstance(self.return_attrs[0], str):
            # 1d attributes, same for each model
            return [[getattr(model, r) for r in self.return_attrs] for model in self.models]
        elif isinstance(self.return_attrs, list) and isinstance(self.return_attrs[0], list):
            # 2d attributes, unique for each model
            return [{r: getattr(model, r) for r in self.return_attrs[i]}
                     for i, model in enumerate(self.models)]
        elif self.return_attrs is None and self.models is not None:
            return self.models
        else:
            return None


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


    def plot(self, npad=2, ax=None, draw_kwargs=None):
        """Plot workflow as a directed graph."""

        if self.graph is None:
            self.create_graph(npad)

        if draw_kwargs is None:
            draw_kwargs = {}
            node_size = draw_kwargs.pop('node_size', 3000)
            font_size = draw_kwargs.pop('font_size', 14)

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        pos = nx.get_node_attributes(self.graph, 'pos')

        nx.draw(self.graph, pos=pos, with_labels=True,
                node_size=node_size, font_size=font_size, **draw_kwargs)


    def create_graph(self, npad=2):
        self.graph = create_graph(self, npad)
