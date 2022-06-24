"""Transformations."""

import numpy as np
from itertools import product


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


    def transform(self, func, *args, axis=None, **kwargs):
        """Queue transformation.

        Parameters
        ----------
        func : function
            Preprocessing function (e.g. filter).
        *args
            Additonal positional arguments to func.
        axis : int or tuple of int, optional, default: None
            Axis to apply the function along. Only used for 2d and greater.
        **kwargs
            Addional keyword arguements to func.
        """
        self.nodes.append(['transform', func, args,
                           {'axis': axis}, kwargs])


    def run_transform(self, func, *args, axis=0, **kwargs):
        """Execute transformation.

        Parameters
        ----------
        func : function
            Preprocessing function (e.g. filter).
        *args
            Additonal positional arguments to func.
        axis : int or tuple of int, optional, default: None
            Axis to apply the function along. Only used for 2d and greater.
        **kwargs
            Addional keyword arguments to func.
        """
        # 1d case
        if self.y_array.ndim == 1:
            self.x_array, self.y_array = func_wrapper(
                func, self.x_array, self.y_array,
                *args, **kwargs
            )
            return

        # Initialize slice indices using numpy-like axis argument
        axis = [axis] if isinstance(axis, int) else axis

        # Invert the axis list to be numpy-like
        axis = tuple([ax for ax in list(range(len(self.y_array.shape)))
                      if ax not in axis])

        inds = [slice(None) if i not in axis else 0
                for i in list(range(len(self.y_array.shape)))]

        # Iterate over axis indices
        mod_shape = None
        for i in product(*[range(i) for i in [self.y_array.shape[i] for i in axis]]):

            # Get slice to pass into func
            for j in range(len(i)):
                inds[axis[j]] = i[j]

            # Infer shape
            if mod_shape is None:

                # Determine if array can be modified in place
                _,  y_array_mod = func_wrapper(
                    func, self.x_array, self.y_array[tuple(inds)],
                    *args, **kwargs
                )

                slice_shape = list(self.y_array[tuple(inds)].shape)
                mod_shape = list(y_array_mod.shape)

                if len(mod_shape) == 0:
                    mod_shape = [1]

                if mod_shape == slice_shape:
                    in_place = True
                else:
                    in_place = False

                    # Determine new shape
                    mod_shape_gen = iter(mod_shape)
                    new_shape = []
                    get_next = True

                    for ind, iax in enumerate(self.y_array.shape):

                        if ind in axis:
                            new_shape.append(iax)
                            continue

                        if get_next:
                            ax_mod = next(mod_shape_gen)

                        if ax_mod == iax:
                            new_shape.append(ax_mod)
                            get_next = True
                        else:
                            new_shape.append(1)
                            get_next = False
                    y_array_reshape = np.zeros(new_shape)

            if not in_place and y_array_mod is not None:
                y_array_reshape[tuple(inds)] = y_array_mod
                y_array_mod = None
            elif in_place and y_array_mod is not None:
                self.y_array[tuple(inds)] = y_array_mod
                y_array_mod = None
            elif not in_place:
                self.x_array, y_array_reshape[tuple(inds)] = func_wrapper(
                    func, self.x_array, self.y_array[tuple(inds)],
                    *args, **kwargs
                )
            else:
                self.x_array, self.y_array[tuple(inds)] = func_wrapper(
                    func, self.x_array, self.y_array[tuple(inds)],
                    *args, **kwargs
                )

        if not in_place:
            self.y_array = np.squeeze(y_array_reshape)


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