"""Models."""

import os
from copy import copy
import numpy as np
import pickle as pkl
from .utils import parse_args, reshape


class Model:
    """Model wrapper.

    Attribues
    ---------
    model : class
        Model class with a .fit method that accepts
        {(x_array, y_array), y_array}.
    nodes : list
        Nodes to append model fitting to.
    """

    def __init__(self, model=None, nodes=None):
        """Initialize model."""
        self.models = []
        self.model = model
        self.nodes = nodes

        if self.nodes is None:
            self.nodes = []

        self.node = None

        if not hasattr(self, 'seeds'):
            self.seeds = None


    def fit(self, model, *args, axis=None, pickle=False, pickle_dir=None, **kwargs):
        """Queue fit.

        Parameters
        ----------
        model : class
            Model class with a .fit method that accepts
            {(x_array, y_array), y_array}.
        args
            Passed to the .fit method of the model class.
        axis : int, optional, default: None
            Axis to fit model over.
        **kwargs
            Passed to the .fit method of the model class.
        """
        self.model = model
        self.nodes.append(['fit', model, args, axis, pickle, pickle_dir, kwargs])


    def run_fit(self, x_array, y_array, *args, axis=None,
                pickle=False, pickle_dir=None, **kwargs):
        """Execute fit.

        Parameters
        ----------
        y_array : ndarray
            Y-axis values. Usually voltage or power.
        x_array : 1d array
            X-axis values. Usually time or frequency.
        *args
            Passed to the .fit method of the model class.
        axis : int, optional, default: None
            Axis to fit model over.
        **kwargs
            Passed to the .fit method of the model class.

        Notes
        -----
        Pass 'self' to any arg or kwarg to infer its value from a instance variable.
        """

        if self.node is not None:
            self.model = self.node[1]
        else:
            self.model = self.nodes[0][1]

        # Get args and kwargs stored in attribute
        args, kwargs = parse_args(list(args), kwargs, self)

        # Apply model to specific axis of y-array
        if axis is not None:
            y_array, _ = reshape(y_array, axis)

            model = []
            for y in y_array:
                _model = copy(self.model)
                if x_array is not None:
                    _model.fit(x_array, y, *args, **kwargs)
                else:
                    _model.fit(y, *args, **kwargs)
                model.append(_model)
            self.model = np.array(model)
        else:
            if x_array is not None:
                self.model.fit(x_array, y_array, *args, **kwargs)
            else:
                self.model.fit(y_array, *args, **kwargs)

        if pickle:

            if pickle_dir is None:
                pickle_dir = './'
            elif not pickle_dir.endswith('/'):
                pickle_dir = pickle_dir + '/'

            if not os.path.isdir(pickle_dir):
                os.mkdir(pickle_dir)

            with open(f"{pickle_dir}/model_{str(self.param_ind).zfill(6)}.pkl", "wb") as f:
                pkl.dump(self.model, f)

        self.models.append(Result(self.model))
        self.model = None

class Result:
    """Class to allow numpy reshaping.

    Notes
    -----
    Numpy sometimes does not like mixed class types in
    object arrays. This prevent invalid __array_struct__
    and allows for easy reshaping of results.
    """
    def __init__(self, result):
        self.result = result
