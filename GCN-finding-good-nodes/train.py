from dataloader import get_data
from copy import deepcopy


def train_node_classifier(model, graphs, dimensions, optimizer, criterion, n_epochs=500):
    for i, data in enumerate(graphs):
        graph, good_nodes = data
        embedding, edge_index, label = get_data(graph, good_nodes, dimensions)
        for epoch in range(1, n_epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(embedding, edge_index)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            acc = eval_node_classifier(model, embedding, edge_index, label)

            if epoch % 10 == 0:
                print(f'Graph: {i:02d}, Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

    return model


def eval_node_classifier(model, att, edge_index, labels):
    model.eval()
    pred = model(att, edge_index).argmax(dim=1)
    correct = (pred == labels).sum()
    acc = int(correct) / len(labels)

    return acc
