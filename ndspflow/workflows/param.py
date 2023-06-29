"""WorkFlow parameterization."""

from inspect import signature
from itertools import product
from functools import partial
from copy import copy, deepcopy

from pathos.multiprocessing import Pool, cpu_count

import numpy as np

from .utils import get_init_params


def run_subflows(wf, iterable, attr, axis=None, n_jobs=-1, progress=None):
    """Parse a workflow's nodes, parameterize (e.g. grid search), and run.

    Parameters
    ---------
    wf : ndspflow.workflows.WorkFlow
        WorkFlow containing ndspflow.workflows.param.Param as (kw)args.
    iterable : list
        Values to pass to each jobs in the pool, specifying either:

        - seeds    : list of int, for simulations
        - subjects : list of int, for reading BIDS directories
        - custom (x_) and y_array : None, instead slices along axis

    attr : {'seeds' or 'subjects'}
        WorkFlow attribue to set iterable to.
    axis : int, optional, default: None
        Axis to iterate over.
    n_jobs : int, optional, default: -1
        Number of jobs to run in parallel.
    progress : {None, tqdm.notebook.tqdm, tqdm.tqdm}
        Progress bar.

    Returns
    -------
    results : 2d, 3d, or 4d list
        Nested results with shape (n_inputs, (n_grid_common,) (n_grid_unique,) n_params).
    """

    # If using an axis
    if axis is not None and iterable is None:
        iterable = np.swapaxes(deepcopy(wf.y_array).T, 0, axis)
        del wf.y_array

    # Split shared and forked nodes
    nodes_common, nodes_unique = parse_nodes(wf.nodes)

    # Compute a param grid and locations to update nodes
    grid_common, locs_common, keys_common = compute_grid(deepcopy(nodes_common))
    nodes_common_grid = nodes_from_grid(deepcopy(nodes_common), grid_common, locs_common)

    grid_unique = []
    keys_unique = []
    nodes_unique_grid = []

    for n in nodes_unique:
        _grid, _locs, _keys = compute_grid(deepcopy(n))
        nodes_unique_grid.append(nodes_from_grid(deepcopy(n), _grid, _locs))
        grid_unique.append(_grid)
        keys_unique.append(_keys)

    pfunc = partial(_run_sub_wf, wf=deepcopy(wf), attr=attr,
                    nodes_common_grid=nodes_common_grid, nodes_unique_grid=nodes_unique_grid)

    if n_jobs == 1:
        # Avoid mp pool
        results = []
        iterable = iterable if progress is None else progress(iterable, total=len(iterable))

        for i in iterable:
            results.append(pfunc(i))

    else:

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(pfunc, iterable)

            if progress is None:
                results = list(mapping)
            else:
                results = list(progress(mapping, total=len(iterable)))

            # Ensure pool exits (needed for pytest cov)
            pool.close()
            pool.join()

    wf.grid_common = grid_common
    wf.grid_unique = grid_unique
    wf.grid_keys_common = keys_common
    wf.grid_keys_unique = keys_unique

    if (wf.nodes[0][0] == 'transform' and
        all([res.shape == results[0][0].shape for res in results[0][1:]])):
        # Transforms will produce stackable arrays
        results = np.stack(results)
        return results

    shape_common = np.squeeze(np.array(grid_common)).shape
    shape_unique = [len(np.unique(i)) for i in
                    np.vectorize(id)(np.squeeze(np.array(grid_unique))).T]

    if np.all(np.array(shape_unique) == shape_unique[0]):
        # Using forks
        results = np.array(results)
    else:
        # Using Param objects
        results = np.array(results, dtype='object')
        results = results.reshape(len(iterable), *list(shape_common) + shape_unique)

    return results


def _run_sub_wf(index, wf=None, attr=None, nodes_common_grid=None, nodes_unique_grid=None):
    """Map-able run function for parallel processing."""

    from .workflow import WorkFlow

    if isinstance(index, np.ndarray):
        wf.y_array = index

    wfs = []

    for nodes_common in nodes_common_grid:

        # Pre fork workflow
        if len(nodes_common) > 0:
            wf_pre = deepcopy(wf)

            if index is not None:
                setattr(wf_pre, attr, index)

            wf_pre.nodes = nodes_common
            wf_pre.run(n_jobs=1)

            ys = wf_pre.y_array
            xs = None
            if wf_pre.x_array is not None:
                xs = wf_pre.x_array
        else:
            ys = wf.y_array
            xs = wf.x_array

        # Post fork workflow
        wfs_sim = []
        seeds = None
        for nodes_post in nodes_unique_grid:

            wfs_fit = []

            for nodes in nodes_post:

                # Re-initialize model after parsing Param object(s)
                for ind in range(len(nodes)):
                    if nodes[ind][0] == 'fit':

                        _params = {attr: getattr(nodes[ind][1], attr) for attr in
                                   get_init_params(nodes[ind][1])}

                        nodes[ind][1].__init__(**_params)
                    elif nodes[ind][0] == 'simulate':
                        seeds = getattr(wf, attr)

                # Create workflow
                wf_param = WorkFlow(seeds)
                wf_param.y_array = ys
                wf_param.x_array = xs
                wf_param.nodes = nodes
                wf_param.run(n_jobs=1)

                if wf_param.results is None and wf_param.x_array is None:
                    # Workflow ended on a transform node.
                    #   Return state of array instead of model results.
                    wfs_fit.append(wf_param.y_array)
                elif wf_param.results is None:
                    wfs_fit.append([wf_param.x_array, wf_param.y_array])
                else:
                    # Model was fit, return results
                    wfs_fit.append(copy(wf_param.results))

            wfs_fit = wfs_fit[0] if len(wfs_fit) == 1 else wfs_fit

            wfs_sim.append(wfs_fit)

        wfs_sim = wfs_sim[0] if len(wfs_sim) == 1 else wfs_sim

        if isinstance(wfs_sim, list) and isinstance(wfs_sim[0], list):
            wfs.append(list(map(list, zip(*wfs_sim))))
        else:
            wfs.append(wfs_sim)

    wfs = wfs[0] if len(wfs) == 1 else wfs

    return wfs


def parse_nodes(nodes):
    """Split nodes into shared and unique.

    Parameters
    ----------
    nodes : list of list
        Nodes defined in WorkFlow.

    Returns
    -------
    nodes_common : list of list
        Prefixed nodes that are shared among all sub-workflows.
    nodes_unique : list of list
        Nodes that are parameterized.
    """
    nodes_unique = []
    nodes_common = []
    nodes_fork = []

    # Get common (shared) nodes
    nodes_common = []

    for i, node in enumerate(nodes):

        if node[0] == 'fork':
            i += 1
            break
        elif node[0] in ['fit', 'transform']:
            break

        if node[0] != 'fork':
            nodes_common.append(node)

    # Get parameterized nodes
    nodes_unique = []
    nodes_fork = []
    nodes_pre_shared = []

    has_fork = False
    forks = {}
    in_fork = None

    for node in nodes[i:]:

        if not has_fork:

            if node[0] == "fork":
                has_fork = True
            else:
                nodes_pre_shared.append(node)
                continue

        if node[0] == 'fork':

            if node[1] not in forks.keys():
                nodes_base = nodes_pre_shared + nodes_fork
                forks[node[1]] = nodes_base
            else:
                nodes_base = forks[node[1]]

            in_fork = node[1]

            nodes_fork = []
        elif node[0] != 'fit':
            nodes_fork.append(node)
        elif node[0] == 'fit':
            if in_fork is not None and in_fork in list(forks.keys()):
                nodes_unique.append(forks[in_fork] + [node])
            else:
                nodes_unique.append([node])

    if len(nodes_unique) == 0 and len(nodes_pre_shared) > 0:
        nodes_unique = [nodes_pre_shared]

    return nodes_common, nodes_unique


def nodes_from_grid(nodes, grid, locs):
    """Use a grid to generate nodes.

    Parameters
    ----------
    nodes : list of list
        Nodes defined in WorkFlow.
    grid : list of list
        All combinations of parameters.
    locs : list of list
        Locations (indices) of nodes that correspond to grid.

    Returns
    -------
    nodes_grid : list of list of list
        Unique sets of nodes from the grid of parameters, with shape
        (n_param_combinations, n_nodes, n_steps_per_node).
    """
    nodes_grid = []

    for params in grid:

        _nodes = deepcopy(nodes)

        kwargs = None

        for param, loc in zip(params, locs):

            # Fit nodes
            if _nodes[loc[0]][0] == 'fit':

                if kwargs is None:

                    # Get model's initalized arguments
                    params_init = get_init_params(_nodes[loc[0]][1])
                    kwargs = {attr: getattr(_nodes[loc[0]][1], attr) for attr in params_init}

                if loc[2] in kwargs:
                    kwargs[loc[2]] = param

            # Non-fit nodes
            if isinstance(_nodes[loc[0]][loc[1]], tuple):
                _nodes[loc[0]][loc[1]] = list(_nodes[loc[0]][loc[1]])

            if isinstance(_nodes[loc[0]][loc[1]], (list, dict)):
                _nodes[loc[0]][loc[1]][loc[2]] = param
                _nodes[loc[0]][loc[1]][loc[2]] = param

        # Re-iniatlize model
        if kwargs is not None:
            _nodes[loc[0]][1].__init__(**kwargs)
            kwargs = None

        nodes_grid.append(_nodes)

    return nodes_grid


def compute_grid(nodes):
    """Compute all combinations of parameterized parameters.

    Parameters
    ----------
    nodes : list of list
        Nodes defined in WorkFlow.

    Returns
    -------
    grid : list of list
        All combinations of parameters.
    locs : list of list
        Locations (indices) of nodes that correspond to grid.
    keys : list of list
        Names of (kw)args.
    """

    locs = []
    grid = []
    keys = []

    for i_node, node in enumerate(nodes):

        if node[0] == 'fit':

            # Get model's initalized arguments
            params_init = get_init_params(node[1])
            p = {attr: getattr(node[1], attr) for attr in params_init}

        for i_step, step in enumerate(node):

            if isinstance(step, tuple):
                step = list(step)

            if isinstance(step, list):
                for i, arg in enumerate(step):
                    if isinstance(arg, Param):
                        locs.append([i_node, i_step, i])
                        grid.append(arg.iterable)

                        _params = list(signature(node[1]).parameters.keys())

                        if node[0] != 'simulate':
                            _params = _params[1:]

                        keys.append(_params[i])

            elif isinstance(step, dict):
                for k, v in step.items():
                    if isinstance(v, Param):
                        locs.append([i_node, i_step, k])
                        grid.append(v.iterable)
                        keys.append(k)

    grid = list(product(*grid, repeat=1))

    return grid, locs, keys


def check_is_parameterized(args, kwargs):
    """Determines if function call is parameterized.

    Parameters
    ----------
    args : tuple
        Function arguments.
    kwargs : dict
        Function keyword arguments.

    Returns
    -------
    is_parameterized : bool
        Whether the call is parameterized.
    """
    is_parameterized = False

    for arg in args:
        if isinstance(arg, Param):
            is_parameterized = True
            break
        for v in kwargs.values():
            if isinstance(v, Param):
                is_parameterized = True
                break

    return is_parameterized


class Param:
    """Smoke class to allow type checking."""

    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        # Dummy iter to not break model classes that
        #   need to iterate over a parameter on initalization
        for i in self.iterable[0]:
            yield i

