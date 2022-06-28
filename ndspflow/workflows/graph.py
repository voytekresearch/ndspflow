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

    # Linear nodes
    xpos = 0
    ypos = 0
    has_input_node = False

    for ind, name in enumerate(node_names):

        if 'read' in name and not has_input_node:
            graph.add_node('read', pos=(xpos, ypos))
            xpos += 1
            last_node = 'read'
            has_input_node = True
        elif 'sim' in name and not has_input_node:
            graph.add_node('sim', pos=(xpos, ypos))
            xpos += 1
            last_node = 'sim'
            has_input_node = True
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
    for k in node_forks:
        if 'read' in node_forks[k]['base']:
            nsplit = node_forks[k]['base'].split('_')
            node_forks[k]['base'] = nsplit[0]
        elif 'simulate' in node_forks[k]['base']:
            nsplit = node_forks[k]['base'].split('ulate')
            node_forks[k]['base'] = nsplit[0] + nsplit[1]
        elif 'transform' in node_forks[k]['base']:
            nsplit = node_forks[k]['base'].split('form')
            node_forks[k]['base'] = nsplit[0] + nsplit[1]

    forks = sorted(list(node_forks.keys()))
    base_shifts = {node_forks[i]['base']: len(node_forks[i]['branch_inds']) for i in node_forks}

    for fork in forks:

        if node_forks[fork]['base'] in list(graph.nodes):
            xpos, ypos = graph.nodes[node_forks[fork]['base']]['pos']
            xpos += 1

        for i, ind in enumerate(node_forks[fork]['branch_inds']):
            base_node = node_forks[fork]['base']
            _xpos = xpos

            for j, lin_ind in enumerate(node_forks[fork]['range'][i]):

                graph.add_node(node_names[lin_ind], pos=(_xpos, ypos))
                if j == 0:
                    graph.add_edge(base_node, node_names[lin_ind])
                    last_node = node_names[lin_ind]
                else:
                    graph.add_edge(last_node, node_names[lin_ind])

                if node_names[lin_ind] in base_shifts.keys():
                    ypos -= base_shifts[node_names[lin_ind]] - 1
                    xpos += 1

                _xpos += 1

            ypos -= 1

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

        if 'read' in name:
            node_names.append('read')
        elif name != 'fork':
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
        if 'read' in name:
            name = 'read'
        elif 'sim' in name:
            name = 'sim'
        if 'fork' in node_names[ind+1] and node_names[ind+1] not in node_forks.keys():
            node_forks[node_names[ind+1]] = {}
            node_forks[node_names[ind+1]]['base'] = name
            node_forks[node_names[ind+1]]['has_base_node'] = False

    # No forking to figure out
    if len(node_forks.keys()) == 0:
        return node_names, None

    # Get indices of branched nodes per fork
    for fork_name in node_forks.keys():
        branch_inds = []
        for ind, name in enumerate(node_names):
            if name == fork_name:
                branch_inds.append(ind+1)

        node_forks[fork_name]['branch_inds'] = branch_inds

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
