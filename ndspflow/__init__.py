# Alias workflow sub-module
from .workflows import (
    WorkFlow,
    BIDS,
    Simulate,
    Transform,
    Model,
    inspect_workflow,
    create_graph,
    run_subflows,
    parse_nodes,
    nodes_from_grid,
    compute_grid,
    check_is_parameterized,
    Param
)
