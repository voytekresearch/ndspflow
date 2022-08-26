"""WorkFlow parameterization."""

from inspect import signature
from itertools import product
from functools import partial
from copy import deepcopy

from multiprocessing import Pool, cpu_count




def run_subflows(wf, iterable, attr, n_jobs=-1, progress=None):
    """Parse a workflow's nodes, parameterize (e.g. grid search), and run.

    Parameters
    ---------
    wf : ndspflow.workflows.WorkFlow
        WorkFlow containing ndspflow.workflows.param.Param as (kw)args.
    iterable : list
        Values to pass to each jobs in the pool, specifying either:

        - seeds    : for simulations
        - subjects : for reading BIDS directories

    attr : {'seeds' or 'subjects'}
        WorkFlow attribue to set iterable to.
    n_jobs : int, optional, default: -1
        Number of jobs to run in parallel.
    progress : {None, tqdm.notebook.tqdm, tqdm.tqdm}
        Progress bar.

    Returns
    -------
    results : list of list of results
        Nested results with shape (n_input_nodes, n_fits, n_params_per_fit).
    """

    # Split shared and forked nodes
    nodes_common, nodes_unique = parse_nodes(wf.nodes)

    # Compute a param grid and locations to update nodes
    grid, locs = compute_grid(deepcopy(nodes_common))

    # Updates nodes for each combination in grid
    nodes_common_grid = nodes_from_grid(deepcopy(nodes_common), grid, locs)

    pfunc = partial(_run_sub_wf, wf=deepcopy(wf), attr=attr,
                     nodes_common_grid=nodes_common_grid, nodes_unique=nodes_unique)

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

    return results


def _run_sub_wf(index, wf=None, attr=None, nodes_common_grid=None, nodes_unique=None):
    """Map-able run function for parallel processing."""

    from .workflow import WorkFlow

    wfs = []

    for nodes_common in nodes_common_grid:

        # Pre fork workflow
        wf_pre = deepcopy(wf)

        if attr == 'seeds':
            setattr(wf_pre, attr, [index])
        else:
            setattr(wf_pre, attr, index)

        wf_pre.nodes = nodes_common
        wf_pre.run(n_jobs=1)

        # Post fork workflow
        sig = wf_pre.y_array

        wfs_sim = []

        for nodes_post in nodes_unique:

            _grid, locs = compute_grid(deepcopy(nodes_post))

            nodes_post = nodes_from_grid(deepcopy(nodes_post), _grid, locs)

            wfs_fit = []

            for _nodes in nodes_post:

                wf_param = WorkFlow()

                wf_param.y_array = sig
                wf_param.nodes = _nodes
                wf_param.run(n_jobs=1)
                wfs_fit.append(wf_param.results)

            wfs_sim.append(wfs_fit)

        wfs.append(wfs_sim)

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

        if node[0] in ['fit', 'transform'] and node[-1]:
            break

        if node[0] != 'fork':
            nodes_common.append(node)

    # Get parameterized nodes
    nodes_unique = []
    nodes_fork = []

    for node in nodes[i:]:
        if node[0] == 'fork' and len(nodes_fork) != 0:
            nodes_unique.append(nodes_fork)
            nodes_fork = []
        else:
            nodes_fork.append(node)

    nodes_unique.append(nodes_fork)

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
    """

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
