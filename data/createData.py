import networkx as nx
import random
import pickle
import argparse


def create_graph(total_nodes, overlap, *args):
    graphs = []

    edges = []
    nodes = [i for i in range(total_nodes)]
    nodes_ac = []
    num_layer = 0
    for item in args:
        num_nodes = item[0]
        num_edges = item[1]
        overlapping_user = int(overlap * len(nodes_ac))

        if num_nodes == 0:
            break
        num_layer += 1

        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))

        random_nodes = random.sample(nodes, num_nodes - overlapping_user)
        if len(nodes_ac) != 0:
            random_nodes_o = random.sample(nodes_ac, overlapping_user)
            random_nodes.extend(random_nodes_o)

        nodes = list(set(nodes) - set(random_nodes))
        nodes_ac.extend(random_nodes)
        nodes_ac = list(set(nodes_ac))

        for i in range(num_edges):
            u, v = random.sample(random_nodes, 2)
            G.add_edge(u, v)

        for i, edge in enumerate(G.edges):
            p = 1 / nx.degree(G)[edge[1]]
            G[edge[0]][edge[1]]['weight'] = p

        for node_ac in G.nodes:
            G.nodes[node_ac]['attribute'] = 1

        edges.extend(G.edges())
        graphs.append(G)

    combined_graph = nx.Graph()
    combined_graph.add_nodes_from(range(total_nodes))
    combined_graph.add_edges_from(edges)
    for i, edge in enumerate(combined_graph.edges):
        p = 1 / nx.degree(combined_graph)[edge[1]]
        combined_graph[edge[0]][edge[1]]['weight'] = p

    for node_ac in combined_graph.nodes:
        combined_graph.nodes[node_ac]['attribute'] = 1

    return graphs, combined_graph, num_layer


parser = argparse.ArgumentParser(description="Create Data")
num_node_1 = [500, 500, 200, 200, 100, 100, 100]
parser.add_argument("-n1", "--num_node1", default=500, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_node_1)))))
num_node_2 = [2000, 1000, 600, 400, 200, 200, 200]
parser.add_argument("-n2", "--num_node2", default=2000, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_node_2)))))
num_node_3 = [2500, 1500, 1000, 600, 400, 300, 300]
parser.add_argument("-n3", "--num_node3", default=2500, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_node_3)))))
num_node_4 = [0, 2000, 1400, 800, 600, 500, 400]
parser.add_argument("-n4", "--num_node4", default=0, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_node_4)))))
num_node_5 = [0, 0, 1800, 1200, 800, 600, 500]
parser.add_argument("-n5", "--num_node5", default=0, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_node_5)))))
num_node_6 = [0, 0, 0, 1800, 1200, 800, 600]
parser.add_argument("-n6", "--num_node6", default=0, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_node_6)))))
num_node_7 = [0, 0, 0, 0, 1700, 1000, 700]
parser.add_argument("-n7", "--num_node7", default=0, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_node_7)))))
num_node_8 = [0, 0, 0, 0, 0, 1500, 800]
parser.add_argument("-n8", "--num_node8", default=0, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_node_8)))))
num_node_9 = [0, 0, 0, 0, 0, 0, 1400]
parser.add_argument("-n9", "--num_node9", default=0, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_node_9)))))

overlap = [0.3, 0.5, 0.7]
parser.add_argument("-o", "--overlapping_user", default=0.3, type=float,
                    help="one of: {}".format(", ".join(str(sorted(overlap)))))

args = parser.parse_args(args=[])

num_edge1 = args.num_node1 * 3
num_edge2 = args.num_node2 * 3
num_edge3 = args.num_node3 * 3
num_edge4 = args.num_node4 * 3
num_edge5 = args.num_node5 * 3
num_edge6 = args.num_node6 * 3
total_nodes = args.num_node1 + args.num_node2 + args.num_node3 + args.num_node4 + args.num_node5 + args.num_node6
print("Num nodes", total_nodes)
graphs, combined_graph, num_layer = create_graph(total_nodes,
                                      args.overlapping_user,
                                      (args.num_node1, num_edge1),
                                      (args.num_node2, num_edge2),
                                      (args.num_node3, num_edge3),
                                      (args.num_node4, num_edge4),
                                      (args.num_node5, num_edge5),
                                      (args.num_node6, num_edge6))

print("Num edge", len(combined_graph.edges))

file_path = "./Data/graph_" + str(total_nodes) + "_node_" + str(num_layer) + "_layer_" + str(
    int(args.overlapping_user * 100)) + "_overlaping_user.pickle"
with open(file_path, 'wb') as file:
    pickle.dump([graphs, combined_graph], file)
