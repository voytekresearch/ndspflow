"""WorkFlow parameterization."""

from inspect import signature
from itertools import product
from functools import partial
from copy import deepcopy

from multiprocessing import Pool, cpu_count

import numpy as np



def run_subflows(wf, seeds, n_jobs=-1, progress=None):
    """Parse a workflows nodes, parameterize (e.g. grid search), and run.

    Paramters
    ---------
    wf : ndspflow.workflows.WorkFlow
        WorkFlow containing ndspflow.workflows.param.Param as (kw)args.

    Returns
    -------
    results : list of list of results
        Nested results with shape (n_input_nodes, n_fits, n_params_per_fit).
    """

    # Split shared and forked nodes
    nodes_common, nodes_unique = parse_nodes(deepcopy(wf.nodes))

    # Compute a param grid and locations to update nodes
    grid, locs = get_grid(deepcopy(nodes_common))

    # Updates nodes for each combination in grid
    nodes_common_grid = nodes_from_grid(deepcopy(nodes_common), grid, locs)

    pfunc = partial(_run_sub_wf, nodes_common_grid=deepcopy(nodes_common_grid),
                    nodes_unique=deepcopy(nodes_unique))

    if n_jobs == 1:

        results = []

        iterable = seeds if progress is None else progress(seeds)

        for seed in iterable:
            results.append(pfunc(seed))

    else:

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(pfunc, seeds)

            if progress is None:
                results = list(mapping)
            else:
                results = list(progress(mapping))

            # Ensure pools exits (needed for pytest cov)
            pool.close()
            pool.join()

    return results


def _run_sub_wf(seed, nodes_common_grid=None, nodes_unique=None):

    from .workflow import WorkFlow

    wfs = []

    for nodes_common in nodes_common_grid:

        # Pre fork workflow
        wf_pre = WorkFlow()

        if seed is not None:
            wf_pre.seeds = [int(seed)]

        wf_pre.nodes = nodes_common
        wf_pre.run(n_jobs=1)

        # Post fork workflow
        sig = wf_pre.y_array

        wfs_sim = []

        for nodes_post in nodes_unique:

            _grid, locs = get_grid(deepcopy(nodes_post))

            nodes_post = nodes_from_grid(deepcopy(nodes_post), _grid, locs)

            wfs_fit = []

            for _nodes in nodes_post:

                wf = WorkFlow()

                wf.y_array = sig
                wf.nodes = _nodes
                wf.run(n_jobs=1)
                wfs_fit.append(wf.results)

            wfs_sim.append(wfs_fit)

        wfs.append(wfs_sim)

    return wfs


def parse_nodes(nodes):

    nodes_unique = []
    nodes_common = []
    nodes_fork = []
    has_common = False

    for node in nodes:

        if node[0] != 'fork' and not has_common:
            nodes_common.append(node)
        elif node[0] == 'fork' and len(nodes_fork) == 0:
            has_common = True
        elif node[0] == 'fork' and len(nodes_fork) != 0:
            nodes_unique.append(nodes_fork)
            nodes_fork = []
        else:
            nodes_fork.append(node)

    nodes_unique.append(nodes_fork)

    return nodes_common, nodes_unique


def get_grid(nodes):

    locs = []
    grid = []

    for i_node, node in enumerate(nodes):

        if node[0] == 'fit':

            p = {attr: getattr(node[1], attr) for attr in
                 list(signature(node[1].__init__).parameters.keys())}

            node = [None, p]

        for i_step, step in enumerate(node):

            if isinstance(step, tuple):
                step = list(step)

            if isinstance(step, list):
                for i, arg in enumerate(step):
                    if isinstance(arg, Param):
                        locs.append([i_node, i_step, i])
                        grid.append(arg.iterable)
            elif isinstance(step, dict):
                for k, v in step.items():
                    if isinstance(v, Param):
                        locs.append([i_node, i_step, k])
                        grid.append(v.iterable)

    grid = list(product(*grid, repeat=1))

    return grid, locs


def nodes_from_grid(nodes, grid, locs):

    nodes_grid = []

    for params in grid:

        _nodes = deepcopy(nodes)

        for param, loc in zip(params, locs):

            if isinstance(_nodes[loc[0]][loc[1]], tuple):
                _nodes[loc[0]][loc[1]] = list(_nodes[loc[0]][loc[1]])

            if isinstance(_nodes[loc[0]][loc[1]], (list, dict)):

                _nodes[loc[0]][loc[1]][loc[2]] = param
                _nodes[loc[0]][loc[1]][loc[2]] = param
            else:
                setattr(_nodes[loc[0]][1], loc[2], param)

        nodes_grid.append(_nodes)

    return nodes_grid


class Param:
    """Smoke class to allow type checking."""

    def __init__(self, iterable):
        self.iterable = iterable
