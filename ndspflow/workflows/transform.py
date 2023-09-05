"""Transformations."""

import numpy as np

from .utils import parse_args, reshape
from .param import check_is_parameterized

class Transform:
    """Transformation class.

    Attributes
    ----------
    y_array : ndarray
        Y-axis values. Usually voltage or power.
    x_array : 1d array, optional, default: None
        X-axis values. Usually time or frequency.
    nodes : list of list
        Contains order of operations as:
        [[function, axis, *args, **kwargs], ...]

    Notes
    -----
    - If x_array is explicitly defined, func is called as func(x_array, y_array).
    - If func returns two parameters, they are set as (x_array, y_array).
    """

    def __init__(self, y_array=None, x_array=None):
        """Initalize object."""

        self.y_array = y_array
        self.x_array = x_array
        self.nodes = []


    def transform(self, func, *args, axis=None, mode=None, **kwargs):
        """Queue transformation.

        Parameters
        ----------
        func : function
            Preprocessing function (e.g. filter).
        *args
            Additonal positional arguments to func.
        axis : int or tuple of int, optional, default: None
            Axis to apply the function along 1d-slices. Only used for 2d and greater.
            Identical to numpy axis arguments. None assumes transform requires 2d input.
        mode : {None, 'notebook'}
            Notebook mode allows functions to be defined in notebooks, rather than
            imported from a module.
        **kwargs
            Addional keyword arguements to func.
        """

        is_parameterized = check_is_parameterized(args, kwargs)
    
        self.nodes.append(['transform', func, args,
                           {'axis': axis}, kwargs, is_parameterized])


    def run_transform(self, func, *args, axis=None, **kwargs):
        """Execute transformation.

        Parameters
        ----------
        func : function
            Preprocessing function (e.g. filter).
        *args
            Additonal positional arguments to func.
        axis : int or tuple of int, optional, default: None
            Axis to apply the function along 1d-slices. Only used for 2d and greater.
            Identical to numpy axis arguments. None assumes transform requires 2d input.
        **kwargs
            Addional keyword arguments to func.

        Notes
        -----
        This is a slightly more flexible/faster version of np.apply_along_axis that
        also handles tuples of axes and can be applied to any series of array operations.
        """

        # Get args and kwargs stored in attributes
        args, kwargs = parse_args(list(args), kwargs, self)

        if axis is not None:

            self.y_array, origshape = reshape(self.y_array, axis)

            _y_array = None

            # Iterate over first axis
            for ind, y in enumerate(self.y_array):

                # Apply function
                x_array, y_array = func_wrapper(func, self.x_array, y, *args, **kwargs)

                # Infer shape compatibility
                if ind == 0 and y_array.shape != y.shape:
                    _y_array = np.zeros((len(self.y_array), *y_array.shape))

                if _y_array is not None:
                    _y_array[ind] = y_array
                elif _y_array is None:
                    self.y_array[ind] = y_array

            # Squeeze and reshape
            if _y_array is None:
                self.y_array = np.squeeze(self.y_array.reshape(*origshape, -1))
            else:
                self.y_array = np.squeeze(_y_array.reshape(*origshape, -1))

            self.x_array = x_array

        else:
            self.x_array, self.y_array = func_wrapper(func, self.x_array, self.y_array,
                                                      *args, **kwargs)


def func_wrapper(func, x_array, y_array, *args, **kwargs):
    """Wrap function to handle variable IO.

    Parameters
    ----------
    func : function
        Transformation function.
    x_array : 1d array
        X-axis definition.
    y_array : 1d array
        Y-axis definition.
    """

    if x_array is None:
        y_array = func(y_array, *args, **kwargs)
    else:
        y_array = func(x_array, y_array, *args, **kwargs)

    if isinstance(y_array, tuple):
        x_array, y_array = y_array
    else:
        x_array = None

    return x_array, y_array