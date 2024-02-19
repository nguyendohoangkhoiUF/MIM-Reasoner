import csv
import os
import pickle
import argparse

from utils_isf import *

parser = argparse.ArgumentParser(description="ISF algorithm")
datasets = ['Data', 'facebook-twitter']
parser.add_argument("-d", "--dataset", default="Data", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
num_node = [600, 5000]
parser.add_argument("-nn", "--num_node", default=600, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_node)))))
num_layer = [3, 4, 5, 6, 7, 8, 9]
parser.add_argument("-nl", "--num_layer", default=3, type=int,
                    help="one of: {}".format(", ".join(str(sorted(num_layer)))))
overlaping_user = [30, 50, 70]
parser.add_argument("-ou", "--overlaping_user", default=30, type=int,
                    help="one of: {}".format(", ".join(str(sorted(overlaping_user)))))
budgets = [10, 20, 30]
parser.add_argument("-b", "--budget", default=30, type=int,
                    help="one of: {}".format(", ".join(str(sorted(budgets)))))

parser.add_argument("-m", "--mc", default=30, type=int,
                    help="the number of Monte-Carlo simulations")
args = parser.parse_args(args=[])

# Read input graph file
file_path = '../Dataset/' + args.dataset + '/graph_' + str(args.num_node) + '_node_' + str(args.num_layer) + \
            '_layer_' + str(args.overlaping_user) + '_overlaping_user.pickle'

file_name = os.path.splitext(os.path.basename(file_path))[0]
with open(file_path, 'rb') as file:
    data = pickle.load(file)
graphs = data[0]
combined_graph = data[1]

# ranking graph
graphs = sorted(graphs, key=lambda graph: (graph.number_of_nodes(), graph.number_of_edges()))

start_time = time.time()
S, SPREAD, timelapse, LOOKUPS, data = celf(combined_graph, k=args.budget, mc=args.mc)
save_time = time.time() - start_time
print("Time", save_time)

################################### Store results to file ###################################

output_path = os.path.join('..', 'Output', file_name + '.csv')
if not os.path.exists(output_path):
    data = [
        ["algorithm", "Nodes", "Edges", "Layer", "Budget", "Seed set", "Spread", "Time"],
        ["ISF", len(combined_graph.nodes), len(combined_graph.edges), len(graphs), args.budget, S, SPREAD[-1], save_time],
    ]
else:
    data = [
        ["ISF", len(combined_graph.nodes), len(combined_graph.edges), len(graphs), args.budget, S, SPREAD[-1], save_time],
    ]

# Save file CSV
with open(output_path, 'a', newline='') as file:
    writer = csv.writer(file)
    for row in data:
        writer.writerow(row)
