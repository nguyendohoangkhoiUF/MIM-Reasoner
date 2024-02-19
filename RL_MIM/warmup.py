from pgm import get_pgm
from utils import *


def get_seedset(combined_graph, graphs, budget_layers, good_nodes, thresold=0.8, mc=10):
    seed_set = []
    layer_pgms = []

    for i, graph in enumerate(graphs):
        for node in graph.nodes:
            if len(layer_pgms) != 0:
                max_prob = layer_pgms[0][node]
                for j in range(len(layer_pgms)):
                    prob = layer_pgms[j][node]
                    if max_prob < prob:
                        max_prob = prob
                graph.nodes[node]['attribute'] = max_prob

        # for node in graph.nodes:
        #     print(graph.nodes[node]['attribute'])

        g_node = good_nodes[i]
        S, SPREAD, timelapse, LOOKUPS, data = celf(graph, budget_layers[i],
                                                   g_node, mc)

        if budget_layers[i] != 0:
            seed_set.extend(S)

        if i < len(graphs) - 1:
            node_pgm = get_pgm(node_status=data, seed_set=S, combined_graph=combined_graph,
                               thresold=thresold)
        layer_pgms.append(node_pgm)

    return seed_set
