"""Workflows."""

from copy import deepcopy
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
from .utils import reshape, extract_results


class WorkFlow(BIDS, Simulate, Transform, Model):
    """Workflow definition.

    Attributes
    ----------
    models : list
        Fit model objects.
    results : list, optional
        Fit model classes or attributes from model classes.
    graph : networkx.DiGraph
        Directed workflow graph.
    nodes : list of list
        Contains order of operations as:
        [[node_type, function, *args, **kwargs], ...]
    params : 3d array
        Parameterized arguments with shape: (n_combs_per_fit, n_fits, n_parameters).
    param_keys : 2d list of str
        Parameterized argument names with shape: (n_fits, n_parameters).
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

        self.params = None
        self.param_keys = None

        # Param grid
        self.grid_common = None
        self.grid_unique = None
        self.grid_keys_common = None
        self.grid_keys_unique = None

    def __call__(self, y_array, x_array=None):
        """Call class to update array inputs."""
        self.y_array = y_array
        if x_array is not None:
            self.x_array = x_array


    def run(self, axis=None, attrs=None, parameterize=False, flatten=False,
            n_jobs=-1, progress=None):
        """Run workflow.

        Parameters
        ----------
        axis : int or tuple of int, optional, default: None
            Axis to pass to multiprocessing pools. Only used for 2d and greater.
            Identical to numpy axis arguments.
        attrs : list of str, optional, default: None
            Model attributes to return.
        parameterize : bool, optional, default: False
            Attempt to parameterize the workflow if True.
        flatten : bool, optional, default: False
            Flattens all models and attributes into a 1d array, per y_array.
        n_jobs : int, optional, default: -1
            Number of jobs to run in parallel.
        progress : {None, tqdm.notebook.tqdm, tqdm.tqdm}
            Progress bar.
        """
        if parameterize:

            from .param import run_subflows

            iterable = None

            for attr in ['seeds', 'subjects']:

                iterable = getattr(self, attr)

                if iterable is not None:
                    break

            self.results = run_subflows(self, iterable, attr, axis=axis,
                                        n_jobs=n_jobs, progress=progress)

            return

        # Handle merges
        _merges = [ind for ind in range(len(self.nodes)) if self.nodes[ind][0] == 'merge']

        if len(_merges) > 0:

            common_nodes = self.nodes[_merges[-1]+1:]
            i_off = 0

            for i in range(len(_merges)):

                # Replace merges with common nodes
                del self.nodes[_merges[i]+i_off]
                i_off -= 1

                for j in range(len(common_nodes)):
                    i_off += 1
                    self.nodes.insert(_merges[i]+i_off, deepcopy(common_nodes[j]))

                # Remove trailing fork node
                if i == len(_merges)-1:
                    self.nodes = self.nodes[:_merges[i]+i_off+1]

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
            _results = self._run((y_array, None), x_array, node_type, flatten)
        elif n_jobs == 1 and self.seeds is None:
            # Don't enter pool if n_jobs is 1 and y_array is 1d
            _results = self._run((y_array, None), x_array=x_array,
                                 node_type=node_type, flatten=flatten)
        else:
            # Parallel execution
            n_jobs = cpu_count() if n_jobs == -1 else n_jobs

            with Pool(processes=n_jobs) as pool:

                mapping = pool.imap(
                    partial(self._run, x_array=x_array, node_type=node_type, flatten=flatten),
                    zip(y_array, np.arange(len(y_array)))
                )

                if progress is not None:
                    _results = list(progress(mapping, total=len(y_array),
                                            desc='Running Workflow'))
                else:
                    _results = list(mapping)

                # Ensure pools exits (needed for pytest cov)
                pool.close()
                pool.join()

        # Workflow ends on transform or sim node
        if self.nodes[-1][0] in ['transform', 'simulate']:
            self.y_array = np.squeeze(np.array(_results))
            return

        # Flatten should return an even array (unless models produce sparse results)
        if flatten:
            self.results = np.array(_results)
            return

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

                # Pull models out of dummy result class
                for inds in product(*[range(i) for i in self.results.shape]):
                    self.results[inds] = self.results[inds].result

            if self.results.ndim == 0:
                self.results = self.results.tolist()

        except (AttributeError, ValueError):
            # Multiple models with unique return shapes are ragged
            #  and resultes are returned as dicts. Don't attempt to reshape.
            pass

        # Reset temporary attributes
        self.model = None
        self.node = None


    def _run(self, input, x_array=None, node_type=None, flatten=False):
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
        flatten : bool, optional, default: False
            Flattens all models and attributes into a 1d array, per y_array.
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
            elif node[0] == 'fit_transform':
                getattr(self, 'run_' + node[0])(*node[2:])
            elif node[0] == 'fork':
                getattr(self, 'run_' + node[0])(node[1])

        # Sort results
        if node[0] in ['simulate', 'transform']:
            return self.y_array

        return extract_results(self.models, self.attrs, flatten)


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


    def merge(self):
        """Queue a merge."""

        self.nodes.append(['merge'])


    def fit_transform(self, model, y_attrs=None, x_attrs=None, axis=None, queue=False,
                      n_jobs=-1, progress=None, fit_args=None, fit_kwargs=None):
        """Queue a model fit and transform y_array to model attributes.

        Parameters
        ----------
        model : class
            Model class with a .fit method that accepts
            {(x_array, y_array), y_array}.
        y_attrs : str or list of str, optional, default: None
            Model attributes to return as y_array.
            Required if model does not have fit_transform method.
        axis : int, optional, default: None
            Axis to fit model over.
        x_attrs : list of str, optional, default: None
            Model attributes to return as x_array.
        n_jobs : int, optional, default: -1
            Number of jobs to run in parallel.
        progress : {None, tqdm.notebook.tqdm, tqdm.tqdm}
            Progress bar.
        queue : bool, optional, default: False
            Add to node queue if True. Otherwise, execute on call.
        fit_args : dict, optional, default: False
            Passed to the .fit method of the model class.
        fit_kwargs : dict, optional, default: False
            Passed to the .fit method of the model class.
        """
        if queue:
            self.nodes.append(['fit_transform', model, y_attrs,  x_attrs,
                               axis, fit_args, fit_kwargs])
        elif hasattr(model, 'fit_transform'):
            self.y_array = model.fit_transform(self.y_array)
        else:
            fit_args = () if fit_args is None else fit_args
            fit_kwargs = {} if fit_kwargs is None else fit_kwargs

            # Fit
            self.fit(model, *fit_args, axis=axis, **fit_kwargs)
            self.run(axis, y_attrs, False, True, n_jobs, progress)

            # Transform
            self.y_array = self.results
            self.x_array = None

            # Clear
            self.results = None
            self.nodes = []
            self.models = []


    def run_fit_transform(self, y_attrs, x_attrs=None, axis=None, args=None, kwargs=None):
        """Execute a fit + transform.

        Parameters
        ----------
        y_attrs : str or list of st
            Model attributes to return as y_array.
        x_attrs : str or list of str, optional, default: None
            Model attributes to return as x_array.
        args : tuple, optional, default: None
            Passed to the .fit method of the model class.
        axis : int, optional, default: None
            Axis to fit model over.
        kwargs : dict, optional, default: None
            Passed to the .fit method of the model class.
        """
        # Fit
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs

        if hasattr(self.node[1], 'fit_transform'):
            self.y_array = self.node[1].fit_transform(self.y_array)
        else:
            self.run_fit(self.x_array, self.y_array, *args, axis=axis, **kwargs)

            # Transform
            self.y_array = extract_results(self.models, y_attrs, True)
            self.x_array = extract_results(self.models, x_attrs, True) if x_attrs is not None else None

        # Clear
        self.results = None
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

    def copy(self):
        return deepcopy(self)