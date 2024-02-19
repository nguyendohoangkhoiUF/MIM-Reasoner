from utils_ksn import *
from multiprocessing import Process, Manager


def profit(graph_i, i, good_node, j, mc_c, results):
    S, SPREAD, timelapse, LOOKUPS, data = celf(graph_i, good_node, j, mc_c)
    results[i] = [S, SPREAD]


def board_generator(graphs, good_nodes, l, mc=1000):
    """
        The board_generator() to query chosen, costs, profits information to use as input for MCKP
        Input:
            graphs: list of graphs
            good_nodes:  good nodes
            l: Budget constraint (integer)
            mc: the number of Monte-Carlo simulations
        Output:
            chosen:  List of seed set for each layer and budget (nested list)
            costs: Cost values for each item in each layer and budget (nested list)
            profits: Profit values for each item in each layer and budget (nested list)
    """

    processes = []

    manager = Manager()
    results = manager.dict()

    for i in range(len(graphs)):
        graph_i = graphs[i]
        p = Process(target=profit, args=(graph_i, i, good_nodes[i], l, mc, results))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    n = len(graphs)
    costs = [[0] * (l + 1) for _ in range(n)]
    profits = [[0] * (l + 1) for _ in range(n)]
    chosen = [[[] for _ in range(l + 1)] for _ in range(n)]

    for i, value in results.items():
        S = value[0]
        SPREAD = value[1]
        for j in range(len(S)):
            chosen[i][j] = S[:(j + 1)]
            costs[i][j] = len(chosen[i][j])
            profits[i][j] = SPREAD[j]

    return chosen, costs, profits
