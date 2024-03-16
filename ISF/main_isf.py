import csv
import os
import pickle
import argparse

from utils_isf import *

parser = argparse.ArgumentParser(description="ISF algorithm")
datasets = ['Xenopus', 'London', 'ObamaInIsrael2013', 'ParisAttack2015', 'Arabidopsis']
parser.add_argument("-d", "--dataset", default="Xenopus", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT', 'SIS']
parser.add_argument("-dm", "--diffusion_model", default="LT", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=5, type=int,
                    help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
mode = ['Normal', 'Budget Constraint']
parser.add_argument("-m", "--mode", default="normal", type=str,
                    help="one of: {}".format(", ".join(sorted(mode))))
args = parser.parse_args(args=[])

file_path = '../Real_Dataset/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG'
with open(file_path, 'rb') as f:
    graphs, multiplex = pickle.load(f)

file_name = os.path.splitext(os.path.basename(file_path))[0]

# ranking graph
graphs = sorted(graphs, key=lambda graph: (graph.number_of_nodes(), graph.number_of_edges()))

start_time = time.time()
budget = int(multiplex.number_of_nodes() * args.seed_rate * 0.01)
S, SPREAD, timelapse, LOOKUPS, data = celf(multiplex, k=budget, diffusion=args.diffusion_model)
save_time = time.time() - start_time
print("Budget", budget)
print("Spread", SPREAD[-1])
print("Time", save_time)

################################### Store results to file ###################################

output_path = os.path.join('..', 'Output', file_name + '.csv')
if not os.path.exists(output_path):
    data = [
        ["algorithm", "Nodes", "Edges", "Layer", "Budget", "Diffusion", "Seed set", "Spread", "Time"],
        ["ISF", len(multiplex.nodes), len(multiplex.edges), len(graphs), args.seed_rate, args.diffusion_model, S, SPREAD[-1], save_time],
    ]
else:
    data = [
        ["ISF", len(multiplex.nodes), len(multiplex.edges), len(graphs), args.seed_rate, args.diffusion_model, S, SPREAD[-1], save_time],
    ]

# Save file CSV
with open(output_path, 'a', newline='') as file:
    writer = csv.writer(file)
    for row in data:
        writer.writerow(row)