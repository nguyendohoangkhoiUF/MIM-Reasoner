import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import copy


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_training_timesteps, device):
        super(ActorCritic, self).__init__()
        self.max_training_timesteps = max_training_timesteps
        self.device = device
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, time_step, best_found_seed_t, mask, mode='duplicate'):
        if mode == 'duplicate':
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            if np.random.rand() < min(time_step / self.max_training_timesteps, 0.8):
                found_seed_t = copy.deepcopy(best_found_seed_t)
                action = torch.tensor(copy.deepcopy(found_seed_t)).to(self.device)
            action_logprob = dist.log_prob(action)
            state_val = self.critic(state)
        elif mode == 'no_duplicate':
            action_probs = self.actor(state)
            action_probs = torch.add(action_probs, 0.000000000001)
            mask_tensor = torch.tensor(mask).to(self.device)
            action_probs = action_probs.masked_fill(mask_tensor > 0, 0)
            action_probs = action_probs / action_probs.sum()
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs + 0.0000001)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


import torch

from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, input_size, hidden_dim1=256, hidden_dim2=256, heads=4):
        super().__init__()
        self.input_size = input_size
        self.gat1 = GATv2Conv(input_size, hidden_dim1, heads=heads)
        self.gat2 = GATv2Conv(hidden_dim1 * heads, hidden_dim2, heads=heads)
        self.gat3 = GATv2Conv(hidden_dim2 * heads, 2, heads=heads)

    def forward(self, x, edge_index):
        x = x.to(torch.float)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.gat3(x, edge_index)
        output = F.log_softmax(x, dim=1)

        return output


import torch
from karateclub import DeepWalk


def get_embeddings(graph, dimensions):
    model = DeepWalk(dimensions=dimensions)
    model.fit(graph)
    embedding = model.get_embedding()
    return embedding


def get_data(graph, dimensions):
    embedding = get_embeddings(graph, dimensions)

    edge_list = list(graph.edges)
    edge_index = torch.tensor(edge_list).t().contiguous()

    return torch.tensor(embedding), edge_index


def eval_node_classifier(model, graph, dimensions=50):
    embedding, edge_index = get_data(graph, dimensions)
    model.eval()
    pred = model(embedding, edge_index).argmax(dim=1)
    good_node = []
    for i, item in enumerate(pred):
        if item == 1:
            good_node.append(i)

    return good_node
