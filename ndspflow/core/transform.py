"""Transformations."""

import numpy as np
from itertools import product


class Transform:
    """Transformation class.

    Attributes
    ----------
    y_arr : ndarray
        Y-axis values. Usually voltage or power.
    x_arr : 1d array, optional, default: None
        X-axis values. Usually time or frequency.
    nodes : list of list
        Contains order of operations as:
        [[function, axis, *args, **kwargs], ...]

    Notes
    -----
    - If x_arr is explicitly defined, func is called as func(x_arr, y_arr).
    - If func returns two parameters, they are set as (x_arr, y_arr).
    """

    def __init__(self, y_arr=None, x_arr=None):
        """Initalize object."""

        self.y_arr = y_arr
        self.x_arr = x_arr
        self.nodes = []


    def transform(self, func, *args, axis=None, **kwargs):
        """Add a node to be excuted with run method.

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


    def run_transform(self, func, *args, axis=None, **kwargs):
        """Apply a preprocessing function along axis.

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
        # 1d case
        if self.y_arr.ndim == 1:
            self.x_arr, self.y_arr = func_wrapper(
                func, self.x_arr, self.y_arr,
                *args, **kwargs
            )

            return

        # Initialize slice indices
        axis = tuple([axis]) if isinstance(axis, int) else axis
        inds = [slice(None) if i not in axis else 0
                for i in list(range(len(self.y_arr.shape)-1))]

        # Iterate over axis indices
        mod_shape = None
        for i in product(*[range(i) for i in [self.y_arr.shape[i] for i in axis]]):

            # Get slice to pass into func
            for j in range(len(i)):
                inds[axis[j]] = i[j]

            # Infer shape
            if mod_shape is None:

                # Determine if array can be modified in place
                _,  y_arr_mod = func_wrapper(
                    func, self.x_arr, self.y_arr[tuple(inds)],
                    *args, **kwargs
                )

                slice_shape = list(self.y_arr[tuple(inds)].shape)
                mod_shape = list(y_arr_mod.shape)

                if mod_shape == slice_shape:
                    in_place = True
                else:
                    in_place = False

                    # Determine new shape
                    mod_shape_gen = iter(mod_shape)
                    new_shape = []
                    get_next = True

                    for ind, iax in enumerate(self.y_arr.shape):

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
                    y_arr_reshape = np.zeros(new_shape)

            if not in_place and y_arr_mod is not None:
                y_arr_reshape[tuple(inds)] = y_arr_mod
                y_arr_mod = None
            elif in_place and y_arr_mod is not None:
                self.y_arr[tuple(inds)] = y_arr_mod
                y_arr_mod = None
            elif not in_place:
                self.x_arr, y_arr_reshape[tuple(inds)] = func_wrapper(
                    func, self.x_arr, self.y_arr[tuple(inds)],
                    *args, **kwargs
                )
            else:
                self.x_arr, self.y_arr[tuple(inds)] = func_wrapper(
                    func, self.x_arr, self.y_arr[tuple(inds)],
                    *args, **kwargs
                )

        if not in_place:
            self.y_arr = np.squeeze(y_arr_reshape)
        

def func_wrapper(func, x_arr, y_arr, *args, **kwargs):
    """Wrap function to handle variable IO.

    Parameters
    ----------
    func : function
        Transformation function.
    x_arr : 1d array
        X-axis definition.
    y_arr : 1d array
        Y-axis definition.
    """

    if x_arr is None:
        y_arr = func(y_arr, *args, **kwargs)
    else:
        y_arr = func(x_arr, y_arr, *args, **kwargs)

    if isinstance(y_arr, tuple):
        x_arr, y_arr = y_arr
    else:
        x_arr = None

    return x_arr, y_arr