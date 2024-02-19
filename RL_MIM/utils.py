import time
import numpy as np
from tqdm import tqdm


def IC(graph, T, mc=100):
    """
        The IC() function describing the spread process
        Input:
            combined_graph: multiplex network
            T: set of seed nodes
            mc: the number of Monte-Carlo simulations
        Output:
            np.mean(spread): average number of nodes influenced by the seed nodes
            np.array(after_activations): set node status
            node_actives: set of active nodes
    """

    spread = []
    after_activations = []
    list_node_actives = []
    for i in range(mc):
        node_actives = set(T[:])
        after_activation = [0] * len(graph.nodes)
        # Simulate propagation process
        new_active, A = T[:], T[:]

        while new_active:  # Enter a while loop that continues until no new nodes are activated in an iteration.

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                # Determine neighbors that become infected
                neighboring_node = [n for n in graph.neighbors(node)]
                neighboring_node = list(set(neighboring_node) - node_actives)
                success = [False] * len(neighboring_node)

                for j, node_nei in enumerate(neighboring_node):

                    p = graph.get_edge_data(node, node_nei)['weight']
                    success[j] = np.random.uniform(0, 1) < p

                new_ones += list(np.extract(success, neighboring_node))
                node_actives.update(new_ones)
            # Update the set of newly activated nodes
            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active
            for activated_node in A:
                after_activation[activated_node] = 1

        # Update the activation status of the seed nodes
        for i, s in enumerate(T):
            after_activation[s] = 2

        spread_x = 0
        for node in A:
            spread_x += graph.nodes[node]['attribute']

        spread.append(spread_x)
        after_activations.append(after_activation)
        list_node_actives.append(node_actives)

    return np.mean(spread), np.array(after_activations), list_node_actives


def celf(graph, k, good_nodes, mc=1000):
    """
        The CELF() function describes the process of finding a seed set
        Input:
            combined_graph: multiplex network
            g_i: graph of the ith layer
            k: budget amount
            mc: the number of Monte-Carlo simulations
        Output:
            S: optimal seed set
            SPREAD: resulting spread
            timelapse: time for each iteration
            LOOKUPS: number of spread calculations
            data: Collection of node status

    """

    # --------------------
    # Find the first node with greedy algorithm
    # --------------------

    # Calculate the first iteration sorted list

    start_time = time.time()
    marg_gain = []
    node_actives = []

    for node in good_nodes:
        s, at, _ = IC(graph, [node], mc)
        marg_gain.append(s)
        node_actives.append(at)

    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(graph.nodes, marg_gain, node_actives), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD, data = [Q[0][0]], Q[0][1], [Q[0][1]], Q[0][2]
    Q, LOOKUPS, timelapse = Q[1:], [len(graph.nodes)], [time.time() - start_time]

    # --------------------
    # Find the next k-1 nodes using the list-sorting procedure
    # --------------------

    for _ in tqdm(range(k - 1)):

        check, node_lookup = False, 0
        if len(Q) > 0:
            while not check:
                # Count the number of times the spread is computed
                node_lookup += 1

                # Recalculate spread of top node
                current = Q[0][0]

                # Evaluate the spread function and store the marginal gain in the list
                s, at, list_node_active = IC(graph, S + [current], mc)

                Q[0] = (current, s - spread, at)

                # Re-sort the list
                Q = sorted(Q, key=lambda x: x[1], reverse=True)

                # Check if previous top node stayed on top after the sort
                check = (Q[0][0] == current)

            # Select the next node
            spread += Q[0][1]
            S.append(Q[0][0])
            # data = np.concatenate((data, Q[0][2]), axis=0)
            data = Q[0][2]
            SPREAD.append(spread)
            LOOKUPS.append(node_lookup)
            timelapse.append(time.time() - start_time)

            # Remove the selected node from the list
            Q = Q[1:]
        else:
            spread = -100
            S.append(S[-1])
            SPREAD.append(spread)
            LOOKUPS.append(node_lookup)
            timelapse.append(time.time() - start_time)

    return S, SPREAD, timelapse, LOOKUPS, data


def IC_rl(combined_graph, T, node_active_graph, mc=100):
    """
        Input:
            combined_graph: multiplex network
            T: set of seed nodes
            mc: the number of Monte-Carlo simulations
        Output:
            np.mean(spread): average number of nodes influenced by the seed nodes
            np.array(after_activations): set node status
            node_actives: set of active nodes
    """

    # Initialize the lists to keep track of activated nodes

    # Loop over the Monte-Carlo simulations

    spread = []
    after_activations = []
    list_node_actives = []

    for i in range(mc):
        node_actives = set(T[:])
        after_activation = [0] * len(combined_graph.nodes)
        # Simulate propagation process
        new_active, A = T[:], T[:]
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                # Determine neighbors that become infected
                neighboring_node = [n for n in combined_graph.neighbors(node)]
                success = [False] * len(neighboring_node)

                for j, node_nei in enumerate(neighboring_node):
                    p = combined_graph.get_edge_data(node, node_nei)['weight']
                    success[j] = np.random.uniform(0, 1) < p

                    for node_i in node_active_graph[i]:
                        if node_i == node_nei:
                            success[j] = np.random.uniform(0, 1) < 0.0
                            break

                new_ones += list(np.extract(success, neighboring_node))
                node_actives.update(new_ones)
            # Update the set of newly activated nodes
            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active
            for activated_node in A:
                after_activation[activated_node] = 1

        # Update the activation status of the seed nodes
        for i, s in enumerate(T):
            after_activation[s] = 2
        sum_spread = 0
        for node_ac in A:
            # print(node_ac, combined_graph.nodes[node_ac]['attribute'])
            sum_spread += combined_graph.nodes[node_ac]['attribute']

        spread.append(sum_spread)
        after_activations.append(activated_node)
        list_node_actives.append(node_actives)

    return np.mean(spread), np.array(after_activations), list_node_actives
