"""Workflows."""

from functools import partial
from itertools import product
from inspect import signature

from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx

from mne_bids import BIDSPath

from .bids import BIDS
from .sim import Simulate
from .transform import Transform
from .model import Model
from .graph import create_graph
from .utils import reshape

class WorkFlow(BIDS, Simulate, Transform, Model):
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

    def __init__(self, y_array=None, x_array=None, **kwargs):
        """Initalize object.

        Parameters
        ----------
        y_array : ndarray, optional, default: None
            Y-axis values. Usually voltage or power.
        x_array : 1d array, optional, default: None
            X-axis values. Usually time or frequency.
        **kwargs
            Additional keyword arguments that sub-classes need access to.
            See: {ndspflow.workflow.bids.BIDS, ndspflow.workflow.sim.Simulate}
        """

        # Parse sub-class initalization kwargs
        bids_kwargs = {}
        bids_kwargs_names = list(signature(BIDS).parameters.keys())[:-1]
        bids_kwargs_names.extend(list(signature(BIDSPath).parameters.keys()))

        sim_kwargs = {}
        sim_kwargs_names = list(signature(Simulate).parameters.keys())

        for k, v in kwargs.items():
            if k in bids_kwargs_names :
               bids_kwargs[k] = v
            elif k in sim_kwargs_names:
                sim_kwargs[k] = v

        # Initialize sub-classes
        Simulate.__init__(self, **sim_kwargs)
        BIDS.__init__(self, **bids_kwargs)

        # Initialize self
        self.nodes = []
        self.models = []

        self.node = None
        self.model = None

        self.graph = None

        self.y_array = y_array
        self.x_array = x_array

        self.y_array_stash = None
        self.x_array_stash = None
        self.fork_inds = []

        self.results = None
        self.attrs = None


    def run(self, axis=None, attrs=None, n_jobs=-1, progress=None):
        """Run workflow.

        Parameters
        ----------
        axis : int or tuple of int, optional, default: None
            Axis to pass to multiprocessing pools. Only used for 2d and greater.
            Identical to numpy axis arguments.
        attrs : list of str, optional, default: None
            Model attributes to return.
        n_jobs : int, optional, default: -1
            Number of jobs to run in parallel.
        progress : {None, tqdm.notebook.tqdm, tqdm.tqdm}
            Progress bar.
        """
        # Reset pre-existing results
        if self.results is not None:
            self.results = None

        if self.fork_inds is not None:
            self.y_array_stash = [None] * len(self.fork_inds)
            self.x_array_stash = [None] * len(self.fork_inds)

        self.attrs = attrs

        # Infer input array type
        origshape = None
        if self.y_array is not None and axis is not None:
            # Drop instance arrays to prevent passing copies to mp pools
            y_array = self.y_array
            self.y_array = None
            x_array = self.x_array
            self.x_array = None

            y_array, origshape = reshape(y_array, axis)

            node_type = 'preload'
        elif self.y_array is not None:
            # Drop instance arrays to prevent passing copies to mp pools
            y_array = self.y_array
            self.y_array = None
            x_array = self.x_array
            self.x_array = None
            node_type = 'preload'
        elif self.seeds is not None:
            # Simulation workflow
            y_array = self.seeds
            x_array = None
            node_type = 'sim'
        elif self.bids_path is not None and self.y_array is None:
            y_array = self.subjects
            x_array = None
            node_type = 'bids'
        else:
            raise ValueError('Undefined input.')

        # Infer if mp is needed
        use_pool = True
        if node_type != 'sim':
            use_pool = not (
                (isinstance(y_array, np.ndarray) and y_array.ndim == 1) |
                (isinstance(y_array, np.ndarray) and axis is None)
            )

        if not use_pool:
            _results = self._run((y_array, 0), x_array, node_type)
        else:
            # Parallel execution
            n_jobs = cpu_count() if n_jobs == -1 else n_jobs

            with Pool(processes=n_jobs) as pool:

                mapping = pool.imap(
                    partial(self._run, x_array=x_array, node_type=node_type),
                    zip(y_array, np.arange(len(y_array)))
                )

                if progress is not None:
                    _results = list(progress(mapping, total=len(y_array),
                                            desc='Running Workflow'))
                else:
                    _results = list(mapping)

        # Workflow ends on transform or sim node
        if self.nodes[-1][0] in ['transform', 'simulate']:
            self.y_array = np.squeeze(np.array(_results))
            return

        if self.results is not None:
            self.results.append(_results)
        else:
            self.results = _results

        try:

            # Squeeze extraneous dimensions
            self.results = np.squeeze(np.array(self.results, dtype='object'))

            if origshape is not None:

                self.results = np.squeeze(self.results.reshape(*origshape, -1))

                # Pull models out of dummy result class
                for inds in product(*[range(i) for i in origshape]):
                    self.results[inds] = self.results[inds].result

            else:
                # Squeeze extraneous dimensions
                self.results = np.squeeze(np.array(self.results, dtype='object'))

                # Pull models out of dummy result class
                for inds in product(*[range(i) for i in self.results.shape]):
                    self.results[inds] = self.results[inds].result

        except AttributeError:
            # Multiple models with unique return shapes  are ragged
            #  and are output as dicts. Don't attempt to reshape.
            pass

        if self.results.ndim == 0:
            self.results = self.results.tolist()

        # Reset temporary attributes
        self.model = None
        self.node = None


    def _run(self, input, x_array=None, node_type=None):
        """Sub-function to allow imap parallelziation.

        Parameters
        ----------
        input : int or 1d array
            Random seed to set if int.
            Signal to process if 1d array.
        x_array : 1d array, optional, default: None
            X-axis values.
        node_type : {None, 'bids', 'sim'}
            Type of node.
        """

        # Reset de-instanced arrays
        self.x_array = x_array

        # Unpack process index
        self.param_ind = input[1]
        input = input[0]

        if node_type == 'sim':
            np.random.seed(input)
        else:
            self.y_array = input

        # Run nodes
        for node in self.nodes:
            self.node = node
            if node[0] == 'read_bids':
                getattr(self, node[0])(self.y_array, node[1], node[2])
            elif node[0] in ['simulate', 'transform']:
                getattr(self, 'run_' + node[0])(node[1], *node[2],
                                                **node[3], **node[4])
            elif node[0] == 'fit':
                getattr(self, 'run_' + node[0])(self.x_array, self.y_array,
                                                *node[2], axis=node[3], **node[4])
            elif node[0] == 'fork':
                getattr(self, 'run_' + node[0])(node[1])
            else:
                getattr(self, 'run_' + node[0])(*node[1], **node[2])

        # Sort results
        if node[0] in ['simulate', 'transform']:
            return self.y_array
        if isinstance(self.attrs, str):
            # Single and same attribute extracted from all models
            return [getattr(model.result, self.attrs) for model in self.models]
        elif isinstance(self.attrs, list) and isinstance(self.attrs[0], str):
            # 1d attributes, same for each model
            return [[getattr(model.result, r) for r in self.attrs] for model in self.models]
        elif isinstance(self.attrs, list) and isinstance(self.attrs[0], list):
            # 2d attributes, unique for each model
            return [{r: getattr(model.result, r) for r in self.attrs[i]}
                     for i, model in enumerate(self.models)]
        elif self.attrs is None and self.models is not None:
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

        if self.y_array_stash[ind] is None:
            # Stash
            self.y_array_stash[ind] = self.y_array.copy()
            if self.x_array is not None:
                self.x_array_stash[ind] = self.x_array.copy()
        elif self.y_array_stash[ind] is not None:
            # Pop
            self.y_array = self.y_array_stash[ind]
            self.x_array = self.x_array_stash[ind]


    def fit_transform(self, model, *args, axis=None, attrs=None,
                      n_jobs=-1, progress=None, **kwargs):
        """Fit a model and transform to the result.

        Parameters
        ----------
        model : class
            Model class with a .fit method that accepts
            {(x_array, y_array), y_array}.
        args
            Passed to the .fit method of the model class.
        axis : int, optional, default: None
            Axis to fit model over.
        attrs : list of str, optional, default: None
            Model attributes to return.
        n_jobs : int, optional, default: -1
            Number of jobs to run in parallel.
        progress : {None, tqdm.notebook.tqdm, tqdm.tqdm}
            Progress bar.
        **kwargs
            Passed to the .fit method of the model class.
        """

        # Queue fit
        self.fit(model, *args, axis=axis, **kwargs)

        # Run Fit
        self.run(attrs=attrs, n_jobs=n_jobs, progress=progress)

        # Transform
        if not isinstance(self.results, np.ndarray):
            raise ValueError('An array must be returned using the attrs argument.')
        else:
            self.y_array = self.results.astype(float)

        # Clear
        self.results = None
        self.nodes = []
        self.models = []


    def drop_x(self):
        """Clear x-array values."""
        self.x_array = None


    def drop_y(self):
        """Clear y-array values."""
        self.y_array = None


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
