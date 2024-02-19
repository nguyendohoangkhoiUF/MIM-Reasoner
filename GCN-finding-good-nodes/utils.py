import time
import numpy as np
import threading
from tqdm import tqdm
import random


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
        node_actives = set(T[:])  # initialized as a set containing the initial seed nodes T.
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

        spread.append(len(A))

    return np.mean(spread), np.array(after_activations), list_node_actives


def celf(graph, k, mc=1000):
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

    Q = [node for node in graph.nodes]
    S = []
    for _ in tqdm(range(k)):
        marg_gain = [0 for node in graph.nodes]
        for node in tqdm(Q):
            s, at, _ = IC(graph, S + [node], mc)
            marg_gain[node] = s

        probabilities = [0 for node in graph.nodes]
        for node in graph.nodes:
            probabilities[node] = marg_gain[node] / np.sum(marg_gain)

        sample_index = random.choices(range(len(probabilities)), probabilities)[0]
        S.append(sample_index)
        Q.remove(sample_index)

    return S


def get_good_node(graph, max_good_node, budget=3, m=4, mc=30):
    gain = [[0 for _ in range(len(graph.nodes))] for _ in range(m)]
    spread = [0 for _ in range(m)]
    for i in range(m):
        S = celf(graph, k=budget, mc=mc)
        spread_prev = 0
        for j in range(len(S)):
            s, at, _ = IC(graph, S[:j + 1], mc)
            gain[i][j] = s - spread_prev
            spread_prev = s

        spread[i] = spread_prev

    scores = {node: 0 for node in graph.nodes}
    sum_spread = np.sum(spread)
    gain_arr = np.array(gain)
    for node in graph.nodes:
        scores[node] = np.sum(gain_arr[:, node]) / sum_spread

    sorted_dict = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Lấy 10 phần tử đầu tiên sau khi đã sắp xếp
    good_nodes = list(dict(sorted_dict[:max_good_node]).keys())

    return good_nodes



