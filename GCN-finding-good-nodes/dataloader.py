import torch
from karateclub import DeepWalk


def get_embeddings(graph, dimensions):
    model = DeepWalk(dimensions=dimensions)
    model.fit(graph)
    embedding = model.get_embedding()
    return embedding


def get_data(graph, good_nodes, dimensions):
    embedding = get_embeddings(graph, dimensions)

    edge_list = list(graph.edges)
    edge_index = torch.tensor(edge_list).t().contiguous()

    label = [0] * len(graph.nodes)
    for node in good_nodes:
        label[node] = 1

    return torch.tensor(embedding), edge_index, torch.tensor(label)
