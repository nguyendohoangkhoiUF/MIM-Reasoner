import torch
from karateclub import DeepWalk
from utils import *
import torch

import torch.nn as nn
from model import GAT
import random
import pandas as pd


def eval_node_classifier(model, att, edge_index, labels):
    model.eval()
    pred = model(att, edge_index).argmax(dim=1)
    correct = (pred == labels).sum()
    acc = int(correct) / len(labels)

    return acc


def get_label(multiplex, graphs, budget_layers):
    labels = [0 for _ in multiplex.nodes]
    for i, graph in enumerate(graphs):
        degrees = {node: 0 for node in graph.nodes}
        for node in graph.nodes():
            degrees[node] = graph.degree(node)

        degrees = list(dict(sorted(degrees.items(), key=lambda x: x[1])))
        for de in degrees[:int(budget_layers[i])]:
            labels[de] = 1

    return labels


def training_good_nodes(multiplex, graphs, budget_layers, dimensions=50, n_epochs=3000, model_path="model.pth"):
    model = DeepWalk(dimensions=dimensions)
    model.fit(multiplex)
    embedding = model.get_embedding()

    labels = get_label(multiplex, graphs, budget_layers)
    embedding = torch.tensor(embedding).to("cuda")
    edge_index = torch.tensor(list(multiplex.edges)).t().contiguous().to("cuda")
    labels = torch.tensor(labels).to("cuda")

    gcn = GAT(input_size=dimensions).to("cuda")
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001, weight_decay=5e-8)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        gcn.train()
        optimizer.zero_grad()
        out = gcn(embedding, edge_index)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = eval_node_classifier(gcn, embedding, edge_index, labels)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

    torch.save(gcn.state_dict(), model_path)


def finding_good_nodes(model_path, multiplex, dimensions=50):
    model = DeepWalk(dimensions=dimensions)
    model.fit(multiplex)
    embedding = model.get_embedding()

    embedding = torch.tensor(embedding).to("cuda")
    edge_index = torch.tensor(list(multiplex.edges)).t().contiguous().to("cuda")

    gcn = GAT(input_size=dimensions).to("cuda")
    gcn.load_state_dict(torch.load(model_path))

    gcn.eval()
    out = gcn(embedding, edge_index)

    indices = list(np.where(out.argmax(dim=1).cpu().numpy() == 1)[0])

    return indices






