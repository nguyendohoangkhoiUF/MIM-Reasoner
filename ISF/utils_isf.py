import time

import numpy as np
from tqdm import tqdm


def IC(graph, T, mc=100):
    """
        The IC() function describing the spread process
        Input:
            graph: graph network
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
        node_actives = set(T[:])  # initialized as a set containing the initial seed nodes T.
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
            # print(new_active)
            A += new_active
            for activated_node in A:
                after_activation[activated_node] = 1

        # Update the activation status of the seed nodes
        for i, s in enumerate(T):
            after_activation[s] = 2

        spread.append(len(A))
        after_activations.append(after_activation)

    return np.mean(spread), np.array(after_activations), list_node_actives


def celf(graph, k, mc=1000):
    """
        The CELF() function describes the process of finding a seed set
        Input:
            graph: graph network
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
    data = None

    for node in tqdm(graph.nodes()):
        s, at, _ = IC(graph, [node], mc)
        marg_gain.append(s)

    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(graph.nodes, marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
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
                s, at, _ = IC(graph, S + [current], mc)
                Q[0] = (current, s - spread)

                # Re-sort the list
                Q = sorted(Q, key=lambda x: x[1], reverse=True)

                # Check if previous top node stayed on top after the sort
                check = (Q[0][0] == current)

            # Select the next node
            spread += Q[0][1]
            S.append(Q[0][0])
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
