"""Create workflow graphs."""


import networkx as nx


def create_graph(wf, npad=2):
    """Create a networkx graph of the workflow.

    Parameters
    ----------
    wf : ndspflow.core.workflow.WorkFlow
        Workflow.
    npad : int, optional, default: 2
        Number of zeros to pad node names with.

    Returns
    -------
    graph : networkx.DiGraph
        Directed workflow graph.
    """

    node_names, node_forks = inspect_workflow(wf, npad)

    graph = nx.DiGraph()

    # Shorten node names
    for ind in range(len(node_names)):
        if 'simulate' in node_names[ind]:
            nsplit = node_names[ind].split('ulate')
            node_names[ind] = nsplit[0] + nsplit[1]
        elif 'transform' in node_names[ind]:
            nsplit = node_names[ind].split('form')
            node_names[ind] = nsplit[0] + nsplit[1]

    for k in node_forks:
        if 'simulate' in node_forks[k]['base']:
            nsplit = node_forks[k]['base'].split('ulate')
            node_forks[k]['base'] = nsplit[0] + nsplit[1]
        elif 'transform' in node_forks[k]['base']:
            nsplit = node_forks[k]['base'].split('form')
            node_forks[k]['base'] = nsplit[0] + nsplit[1]

    # Linear nodes
    xpos = 0
    ypos = 0
    has_sim_node = False

    for ind, name in enumerate(node_names):

        if 'sim' in name and not has_sim_node:
            graph.add_node('sim', pos=(xpos, ypos))
            xpos += 1

            last_node = 'sim'
            has_sim_node = True
        ## Add conditional here for non-simulation based workflows
        elif 'fork' in name:
            # Fork logic is dealt with below
            break
        elif 'sim' not in name:


            graph.add_node(name, pos=(xpos, ypos))
            graph.add_edge(last_node, name)
            xpos += 1
            last_node = name

    # Without forking
    if node_forks is None:
        return graph

    # Withforking
    forks = sorted(list(node_forks.keys()))
    for fork in forks:

        if node_forks[fork]['base'] in list(graph.nodes):
            xpos, ypos = graph.nodes[node_forks[fork]['base']]['pos']
            xpos += 1

        _ypos = ypos
        for i, ind in enumerate(node_forks[fork]['branch_inds']):

            base_node = node_forks[fork]['base']
            _xpos = xpos

            for j, lin_ind in enumerate(node_forks[fork]['range'][i]):
                graph.add_node(node_names[lin_ind], pos=(_xpos, _ypos))
                if j == 0:
                    graph.add_edge(base_node, node_names[lin_ind])
                    last_node = node_names[lin_ind]
                else:
                    graph.add_edge(last_node, node_names[lin_ind])
                _xpos += 1

            _ypos -= 1 + node_forks[fork]['n_nodes'][i]

    return graph


def inspect_workflow(wf, npad=2):
    """Infers nodes and edges from a workflow.

    Parameters
    ----------
    wf : ndspflow.core.workflow.WorkFlow
        Workflow.
    npad : int, optional, default: 2
        Number of zeros to pad node names with.

    Returns
    -------
    node_names : list or str
        Names of all nodes.
    node_forks : dict
        Metadata for sub-workflows.
    """

    # Get node names
    node_names = []
    inds = {}
    for node in wf.nodes:
        name = node[0]
        if name not in inds.keys():
            inds[name] = 0
        else:
            inds[name] += 1

        if name != 'fork':
            node_names.append(name + str(inds[name]).zfill(npad))
        else:
            node_names.append(name + str(node[1]).zfill(npad))

    # Get base nodes prior to forks
    node_forks = {}
    for ind, name in enumerate(node_names):

        # Prevent index error
        in_range = ind < len(node_names)-1

        if not in_range:
            break

        # Determine nodes before forks
        if 'fork' in node_names[ind+1] and node_names[ind+1] not in node_forks.keys():
            node_forks[node_names[ind+1]] = {}
            node_forks[node_names[ind+1]]['base'] = name
            node_forks[node_names[ind+1]]['has_base_node'] = False

    # No forking to figure out
    if len(node_forks.keys()) == 0:
        return node_forks, None

    # Get indices of branched nodes per fork
    for fork_name in node_forks.keys():
        branch_inds = []
        for ind, name in enumerate(node_names):
            if name == fork_name:
                branch_inds.append(ind+1)

        node_forks[fork_name]['branch_inds'] = branch_inds
        node_forks[fork_name]['n_nodes'] = [0] * len(branch_inds)

    # Get number of nodes per fork
    base_names = {node_forks[i]['base']:i for i in node_forks}
    for fork_name in node_forks.keys():
        n_desc = [0] * len(node_forks[fork_name]['branch_inds'])
        for i, ind in enumerate(node_forks[fork_name]['branch_inds']):
            if node_names[ind] in base_names.keys():
                n_branches = len(node_forks[base_names[node_names[ind]]]['branch_inds'])
                node_forks[fork_name]['n_nodes'][i] = n_branches

    # Get number of additional linear nodes
    node_inds = {}
    inds = sorted([i for n in node_forks
                   for i in node_forks[n]['branch_inds']])

    for i, ind in enumerate(inds):
        if i < len(inds)-1:
            diff = inds[i+1] - ind - 1
            node_inds[ind] = range(ind, ind+diff)
        else:
            node_inds[ind] = range(ind, len(node_names))

    for i in node_forks:
        node_forks[i]['range'] = [node_inds[j] for j in node_forks[i]['branch_inds']]

    return node_names, node_forks
