"""WorkFlow parameterization."""

from inspect import signature
from itertools import product
from copy import deepcopy
from re import sub

import numpy as np


def parameterize_workflow(wf):
    """Parse a workflows nodes and parameterizes (e.g. grid search).

    Paramters
    ---------
    wf : ndspflow.workflows.WorkFlow
        WorkFlow containing Param (kw)args.

    Returns
    -------
    wf : ndspflow.workflows.WorkFlow
        Expanded workflow.

    Notes
    -----
    The search for the Param class only goes one level deep on
    .fit nodes, meaning Model(thresh=Param([.5, .8])) will work,
    but Model(thresh={'amp': Param([.5, .8])}) will not.
    """
    # Parse nodes for Param class
    grid = []
    locs = []

    for i, node in enumerate(wf.nodes):

        offset = 1

        if node[0] in ['simulate', 'transform']:
            offset += 1

        for j, p in enumerate(node[offset:]):

            # Parse model's init for Param
            if node[0] == 'fit' and j == 0:
                p = {attr: getattr(node[1], attr) for attr in
                     list(signature(node[1].__init__).parameters.keys())}

            # Ensure mutable
            if isinstance(p, tuple):
                p = list(p)

            # Parse args and kwargs for Param
            if isinstance(p, dict):
                for k, v in p.items():
                    if isinstance(v, Param):
                        grid.append(v.iterable)
                        locs.append([i, j+offset, k])
            elif isinstance(p, (tuple, list, np.ndarray)):
                for ki, k in enumerate(p):
                    if isinstance(k, Param):
                        grid.append(k.iterable)
                        locs.append([i, j+offset, ki])

    # Create parameter grid
    grid = list(product(*grid, repeat=1))


    # Re-create nodes
    fork_ind = locs[:][0][0]
    nodes = wf.nodes[:fork_ind]

    for i, params in enumerate(grid):

        sub_nodes = deepcopy(wf.nodes)

        for j, loc in enumerate(locs):
            if sub_nodes[loc[0]][0] == 'fit' and loc[1] == 1:
                setattr(sub_nodes[loc[0]][1], loc[2], params[j])
            else:
                sub_nodes[loc[0]][loc[1]][loc[2]] = params[j]

        nodes.extend(sub_nodes[fork_ind:])

        if i != len(grid)-1:
            nodes.append(['fork', -1])

    # Insert inital work such that all params in the grid
    #   share common pre-nodes
    if fork_ind > 0:
        nodes.insert(fork_ind, ['fork', -1])

    wf.nodes = nodes
    wf.fork_inds.insert(0, -1)

    # Track & reshape parameters
    base = []
    pinds = []

    param_keys = []
    param_inds = []

    # Get keys of parameters, reshaped with unique/single fits
    for ind, loc in enumerate(locs):

        if sub_nodes[loc[0]][0] in ['transform', 'simulate']:

            if isinstance(loc[2], str):
                base.append(loc[2])
            else:
                base.append(
                    list(signature(sub_nodes[loc[0]][1]).parameters.keys())[loc[2]]
                )

            pinds.append(ind)

        elif sub_nodes[loc[0]][0] in ['fit', 'fit_transform']:

            _base = base.copy()
            _pinds = pinds.copy()

            if isinstance(loc[2], str):
                _base.append(loc[2])
            else:
                _base.append(list(signature(loc[1].fit).parameters.keys())[loc[2]])

            _pinds.append(ind)

            param_inds.append(_pinds)
            param_keys.append(_base)

    # Reshape grid and results
    params = []

    for ps in grid:
        _params = [[], []]
        j = 0

        for inds, keys in zip(param_inds, param_keys):

            for i, k in zip(inds, keys):
                _params[j].append(ps[i])

            j += 1

        params.append(_params)

    params = np.array(params, dtype='object')

    wf.params = params
    wf.param_keys = param_keys

    return wf


class Param:
    """Smoke class to allow type checking."""

    def __init__(self, iterable):
        self.iterable = iterable
