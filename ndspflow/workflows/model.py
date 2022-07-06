"""Models."""

from copy import copy
import numpy as np

from .utils import reshape


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
        self.model = model
        self.nodes = nodes
        self.attrs = None


    def fit(self, model, *args, axis=None, **kwargs):
        """Queue fit."""
        self.model = model
        self.nodes.append(['fit', model, args, axis, kwargs])


    def run_fit(self, x_array, y_array, *args, axis=None, **kwargs):
        """Execute fit.

        Parameters
        ----------
        y_array : ndarray, optional, default: None
            Y-axis values. Usually voltage or power.
        x_array : 1d array, optional, default: None
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

        self.model = self.node[1]

        if isinstance(self.attrs, str):
            self.attrs = [self.attrs]

        # Get args and kwargs stored in attributes
        args = list(args)

        for ind in range(len(args)):
            if isinstance(args[ind], str) and 'self' in args[ind]:
                args[ind] = getattr(self, args[ind].split('.')[-1])

        for k, v in kwargs.items():
            if isinstance(v, str) and 'self' in v:
                kwargs[k] = getattr(self, v.split('.')[-1])

        # Fit follwoing merge assumes current state of y-array is required
        if hasattr(self, '_merged_fit') and self._merged_fit:
            self.model.fit(y_array, *args, **kwargs)
            self.results = self.model
            self.model = None
            return

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
            self.model = model
        else:
            if x_array is not None:
                self.model.fit(x_array, y_array, *args, **kwargs)
            else:
                self.model.fit(y_array, *args, **kwargs)

        self.models.append(Result(self.model))
        self.model = None

class Merge:
    """Dummy model used to merge arrays.

    Notes
    -----
    The .fit method is a pass-through that collects
    the y-array into a temporary attribute.
    """

    def __init__(self):
        self._x_array = None
        self._y_array = None

    def fit(self, *args):

        if len(args) == 1:
            self._y_array = args[0]
        elif len(args) == 2:
             self._x_array = args[0]
             self._y_array = args[1]

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
