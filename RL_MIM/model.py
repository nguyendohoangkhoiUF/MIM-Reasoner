import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import copy
import torch
from torch.nn import Dropout
from torch_geometric.nn import GATv2Conv, GCNConv
import torch.nn.functional as F
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_training_timesteps, device):
        super(ActorCritic, self).__init__()
        self.max_training_timesteps = max_training_timesteps
        self.device = device
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
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
            # action_probs = self.actor(state)
            # action_probs = torch.add(action_probs, 0.000000000001)
            # mask_tensor = torch.tensor(mask).to(self.device)
            # action_probs = action_probs.masked_fill(mask_tensor > 0, 0)
            # action_probs = action_probs / action_probs.sum()
            # dist = Categorical(action_probs)
            # action = dist.sample()
            # action_logprob = dist.log_prob(action)
            # state_val = self.critic(state)
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            if np.random.rand() < min(time_step / self.max_training_timesteps, 0.8):
                found_seed_t = copy.deepcopy(best_found_seed_t)
                action = torch.tensor(copy.deepcopy(found_seed_t)).to(self.device)
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


class GAT(torch.nn.Module):
    def __init__(self, input_size, hidden_dim1=128, hidden_dim2=128):
        super().__init__()
        self.input_size = input_size
        self.conv1 = GCNConv(input_size, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, 2)

    def forward(self, x, edge_index):
        x = x.to(torch.float)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        output = F.log_softmax(x, dim=1)

        return output