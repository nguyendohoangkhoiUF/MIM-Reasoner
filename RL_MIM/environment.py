import torch
from utils import *
from copy import  deepcopy


class GraphDreamerEnv(object):
    def __init__(self, combined_graph, budget, S, support_decay, spread, mc=1):
        self.combined_graph = combined_graph  # multiplex network
        self.budget = budget  # budget amount
        self.mc = mc  # the number of Monte-Carlo simulations

        self.num_nodes_graph = len(self.combined_graph.nodes)  # number of nodes
        self.mask = None  # The mask marks the selected nodes
        self.seed_node = set()  # Collection of seed nodes
        self.done = None  # Check out the end of the episode
        self.state = None  # The state is represented as an array with the number of elements equal to the budget provided, where each element corresponds to a selected seed node.
        self.num_step = None  # number of steps
        self.reward = None  # reward
        self.support_decay = support_decay  #0.999999
        self.activation_node = [[] for _ in
                                range(self.mc)]  # The status of the activated nodes in each Monte Carlo simulation run.
        self.maximum_reward = spread  # Biggest Reward currently.
        self.best_action = S  # Best action sequence currently.
        self.previous_spread = 0  # The extent of the previous spread.
        self.best_spread_of_this_step = torch.zeros(self.budget)  # Best spread of this step.
        self.best_action_so_far = torch.zeros(self.budget)
        self.support_factor = 100

    def reset(self, mode):
        if mode == 'change_graph':
            self.combined_graph = None

        self.mask = torch.zeros(len(self.combined_graph.nodes))
        self.state = torch.zeros(self.budget + 1)
        self.seed_node = set()
        self.done = False
        self.num_step = 0
        self.reward = 1
        self.previous_spread = 0
        self.activation_node = [[] for _ in range(self.mc)]
        return self.state, self.mask, self.done

    def step(self, action):
        assert not self.done
        A = []
        current_spread = 0
        if self.num_step + 1 >= self.budget:
            self.done = True
        self.mask[action] = 2
        self.state[self.num_step] = action
        self.state[-1] = self.num_step
        self.num_step += 1

        self.seed_node.add(action)
        S = list(self.seed_node)

        adj_matrix = nx.to_scipy_sparse_array(self.combined_graph, dtype=np.float32, format='csr')
        current_spread, A = diffusion_evaluation(adj_matrix, S)

        # current_spread, at, A = IC_rl(deepcopy(self.combined_graph), S, self.activation_node, mc=self.mc)
        self.activation_node = A

        # if 0.1 < self.support_factor:
        #     support_mode = 1
        #     self.support_factor = self.support_factor*self.support_decay
        # else:
        #     support_mode = 0
        #
        # if support_mode == 1:
        #     self.reward = - ((action - self.best_action[self.num_step - 1]) / self.num_nodes_graph) ** 2
        #     if action == self.best_action[self.num_step - 1]:
        #         self.reward += 0.1 * self.num_nodes_graph
        # else:
        if self.num_step == self.budget:
            self.reward = current_spread
        self.previous_spread = current_spread
        # if current_   spread >= self.best_spread_of_this_step[
        #     self.num_step - 1] and action not in self.best_action_so_far[
        #                                          :self.num_step - 1]:
        #     self.best_spread_of_this_step[self.num_step - 1] = current_spread
        #     self.reward += current_spread
        #     self.best_action_so_far[self.num_step - 1] = action

        return self.state, self.reward, self.done, self.mask, current_spread
