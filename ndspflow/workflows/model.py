"""Models."""

from copy import copy
from inspect import signature

from .utils import parse_args, reshape, get_init_params
from .param import Param

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

        self.params_init = None


    def fit(self, model, *args, axis=None, **kwargs):
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

        # Determine if any Param objects have been passed to model initalization
        is_parameterized = False

        self.params_init = get_init_params(model)

        for p in self.params_init:
            if isinstance(getattr(model, p), Param):
                is_parameterized = True
                break

        self.nodes.append(['fit', model, args, axis, kwargs, is_parameterized])


    def run_fit(self, x_array, y_array, *args, axis=None, **kwargs):
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

            models = []
            for y in y_array:
                _model = copy(self.model)
                if x_array is not None:
                    mfit = _model.fit(x_array, y, *args, **kwargs)
                else:
                    mfit = _model.fit(y, *args, **kwargs)

                # Some model's .fit method returns a results object (e.g. statmodels)
                #   Other libraries (e.g. sklearn) update results in self.
                if mfit is None:
                    models.append(_model)
                else:
                    models.append(mfit)

            models = [Result(m) for m in models]

            if len(self.models) != 0 or self.models is None:
                self.models = [self.models, models]
            else:
                self.models = models
        else:
            if x_array is not None:
                mfit = self.model.fit(x_array, y_array, *args, **kwargs)
            else:
                mfit = self.model.fit(y_array, *args, **kwargs)

            # Some model's .fit method returns a results object (e.g. statmodels)
            #   Other libraries (e.g. sklearn) update results in self.
            if mfit is None:
                self.models.append(Result(self.model))
            else:
                self.models.append(Result(mfit))

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
