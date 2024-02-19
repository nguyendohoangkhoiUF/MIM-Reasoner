import networkx as nx
import random
import pickle
from utils import *


def create_graph(lowerNodeLimit, upperNodeLimit, lowerEdgeLimit, upperEdgeLimit, num_graph):
    graphs = []
    for i in range(num_graph):
        num_nodes = random.randint(lowerNodeLimit, upperNodeLimit)
        num_edges = random.randint(lowerEdgeLimit, upperEdgeLimit)
        graph = nx.gnm_random_graph(num_nodes, num_edges)
        for i, edge in enumerate(graph.edges):
            p = 1 / nx.degree(graph)[edge[1]]
            graph[edge[0]][edge[1]]['weight'] = p

        max_good_node = min(int(0.3 * len(graph.nodes)),1000)
        budget = 30
        good_nodes = get_good_node(graph, max_good_node, budget=budget, m=30, mc=20)
        graphs.append([graph, good_nodes])

    return graphs


graphs = create_graph(1000, 2000, 3000, 6000, 100)
with open("graphs.pickle", 'wb') as file:
    pickle.dump(graphs, file)


