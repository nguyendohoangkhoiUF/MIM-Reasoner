from pgm import get_pgm
from utils import *

def get_seedset(combined_graph, graphs, good_nodes, budget_layers, thresold=0.8, diffusion='LT'):
    seed_set = []
    layer_pgms = []
    
    for i, graph in enumerate(graphs):
        for node in combined_graph.nodes:
            if len(layer_pgms) > 0:
                min_prob = layer_pgms[0][node]
                for j in range(len(layer_pgms)):
                    prob = layer_pgms[j][node]
                    if min_prob > prob:
                        min_prob = prob
                combined_graph.nodes[node]['attribute'] = min_prob

        S, SPREAD, _, _, data = celf(combined_graph, graph, good_nodes, budget_layers[i], diffusion)
        
        if budget_layers[i] > 0:
            seed_set.extend(S)
            
        if i < len(graphs) - 1:
            node_pgm = get_pgm(node_status=data, seed_set=S, combined_graph=combined_graph,
                               thresold=thresold)
            layer_pgms.append(node_pgm)
            
            for node_x in S:
                node_pgm[node_x] = 0.0

    return seed_set
