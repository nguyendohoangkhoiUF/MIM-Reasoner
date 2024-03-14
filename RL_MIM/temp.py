import csv
import os
import pickle
from model import *

from agent import PPO
from board_generator import board_generator
from environment import GraphDreamerEnv
from mckp import mckp_constraint_solver
from utils import *
from warmup import get_seedset
import argparse
import warnings
from copy import deepcopy
from gcn_good_nodes import finding_good_nodes, training_good_nodes
from model import *
import random

parser = argparse.ArgumentParser(description="KSN algorithm")
parser = argparse.ArgumentParser(description="ISF algorithm")
datasets = ['Xenopus', 'London', 'ObamaInIsrael2013', 'ParisAttack2015', 'Arabidopsis']
parser.add_argument("-d", "--dataset", default="Xenopus", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT', 'SIS']
parser.add_argument("-dm", "--diffusion_model", default="LT", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=20, type=int,
                    help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
mode = ['Normal', 'Budget Constraint']
parser.add_argument("-m", "--mode", default="normal", type=str,
                    help="one of: {}".format(", ".join(sorted(mode))))

parser.add_argument("-t", "--thresold", default=0.8, type=float,
                    help="Clustering threshold")

parser.add_argument("-spd", "--support_decay", default=0.999999, type=float,
                    help="support factor decay")

parser.add_argument("-tr", "--training", default=False, type=bool,
                    help="Training")

# RL-Agent
parser.add_argument("-ep", "--epochs", default=40, type=int, help="K-epochs")
parser.add_argument("-eps", "--eps_clip", default=0.2, type=float, help="Epsilon clip")
parser.add_argument("-ga", "--gamma", default=1, type=int, help="Gamma")
parser.add_argument("-lra", "--lr_actor", default=0.0001, type=float, help="Learning rate actor")
parser.add_argument("-lrc", "--lr_critic", default=0.001, type=float, help="Learning rate critic")

args = parser.parse_args(args=[])

file_path = '../data/' + args.dataset + '_mean_' + args.diffusion_model + str(10 * args.seed_rate) + '.SG'
with open(file_path, 'rb') as f:
    graphs, multiplex = pickle.load(f)


for _ in range(100):
    S = random.sample(list(multiplex.nodes()), 10)
    adj_matrix = nx.to_scipy_sparse_array(multiplex, dtype=np.float32, format='csr')
    current_spread, A = diffusion_evaluation(adj_matrix, S)
    print(S, current_spread)
