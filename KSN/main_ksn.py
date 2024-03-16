from utils_ksn import *
import pickle
from mckp import mckp_constraint_solver
from board_generator import board_generator
import os
import csv
import time
import argparse


if __name__ == '__main__':
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

    for node in multiplex.nodes():
        multiplex.nodes[node]['attribute'] = 1
        
    start_time = time.time()
    budget = int(multiplex.number_of_nodes() * args.seed_rate * 0.01)
    chosen, costs, profits = board_generator(multiplex, graphs, l=budget, diffusion=args.diffusion_model)
    seed_set, results = mckp_constraint_solver(len(graphs), chosen, costs, profits, l=budget)
    
    adj_matrix = nx.to_scipy_sparse_array(multiplex, dtype=np.float32, format='csr')
    spread, after_activations = diffusion_evaluation(adj_matrix, set(seed_set), diffusion=args.diffusion_model)
        
    save_time = time.time() - start_time
    print("Time", save_time)
    print("Seed set", seed_set)
    print("Budget", budget)
    print("Spread", spread)

    ################################### Store results to file ###################################

    output_path = os.path.join('..', 'Output', file_name + '.csv')

    if not os.path.exists(output_path):
        data = [
            ["algorithm", "Nodes", "Edges", "Layer", "Budget", "Diffusion", "Seed set", "Spread", "Tỉme"],
            ["KSN", len(multiplex.nodes), len(multiplex.edges), len(graphs), args.seed_rate, args.diffusion_model, seed_set, spread,
             save_time]
        ]
    else:
        data = [
            ["KSN", len(multiplex.nodes), len(multiplex.edges), len(graphs), args.seed_rate, args.diffusion_model, seed_set, spread,
             save_time]
        ]

    # Save file CSV
    with open(output_path, 'a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
