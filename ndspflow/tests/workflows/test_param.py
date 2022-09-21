"""Tests for workflow parameterization."""

import pytest

from ndspflow.workflows.param import (run_subflows, parse_nodes,
    nodes_from_grid, compute_grid, check_is_parameterized, Param)

@pytest.mark.parametrize('args_paramed', (True, False))
@pytest.mark.parametrize('kwargs_paramed', (True, False))
def test_check_is_parameterized(args_paramed, kwargs_paramed):

    args = [0, 1, 2]
    kwargs = {'x': 0, 'y': 1}

    if args_paramed:
        args[0] = Param([-1, 0])

    if kwargs_paramed:
        kwargs['x'] = Param([-1, 0])

    if args_paramed or kwargs_paramed:
        assert check_is_parameterized(args, kwargs)
    else:
        assert not check_is_parameterized(args, kwargs)
