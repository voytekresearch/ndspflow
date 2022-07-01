"""Models."""

from copy import copy
from inspect import signature


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

        self.model_self = None
        self.return_attrs = None


    def fit(self, model, *args, **kwargs):
        """Queue fit."""
        self.model = model
        self.nodes.append(['fit', model, args, kwargs])


    def run_fit(self, x_array, y_array, *args, **kwargs):
        """Execute fit.

        Parameters
        ----------
        y_array : ndarray, optional, default: None
            Y-axis values. Usually voltage or power.
        x_array : 1d array, optional, default: None
            X-axis values. Usually time or frequency.
        *args
            Passed to the .fit method of the model class.
        **kwargs
            Passed to the .fit method of the model class.

        Notes
        -----
        Pass 'self' to any arg or kwarg to infer its value from a instance variable.
        """
        self.model = self.node[1]

        if isinstance(self.return_attrs, str):
            self.return_attrs = [self.return_attrs]

        # Get args and kwargs stored in attributes
        args = list(args)

        for ind in range(len(args)):
            if isinstance(args[ind], str) and 'self' in args[ind]:
                args[ind] = getattr(self, args[ind].split('.')[-1])

        for k, v in kwargs.items():
            if isinstance(v, str) and 'self' in v:
                kwargs[k] = getattr(self, v.split('.')[-1])

        # Models expect 1d array inputs
        if y_array.ndim >= 2:
            y_array = y_array.reshape(-1, y_array.shape[-1])
        else:
            y_array = y_array.reshape(1, y_array.shape[0])

        for y in y_array:
            _model = copy(self.model)
            if x_array is not None:
                _model.fit(x_array, y, *args, **kwargs)
            else:
                _model.fit(y, *args, **kwargs)

            self.models.append(_model)


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
