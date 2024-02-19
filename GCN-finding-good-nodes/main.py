import torch
from model import GAT
import torch.nn as nn
from train import train_node_classifier
import pickle
import random
import networkx as nx

# Read input graph file
file_path = "graphs.pickle"

with open(file_path, 'rb') as file:
    graphs = pickle.load(file)

# graphs_add = graphs[:3]
# for i in range(len(graphs)*1):
#     graphs_add = graphs_add + graphs
# random.shuffle(graphs_add)


dimensions = 50
mc = 30
gcn = GAT(input_size=dimensions).to("cpu")
optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.001, weight_decay=5e-8)
criterion = nn.CrossEntropyLoss()

gcn = train_node_classifier(gcn, graphs, dimensions, optimizer_gcn, criterion)
path = "model.pth"
torch.save(gcn.state_dict(), path)

