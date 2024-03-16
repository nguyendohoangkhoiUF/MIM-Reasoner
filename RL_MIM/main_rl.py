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

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KSN algorithm")
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

    parser.add_argument("-t", "--thresold", default=0.8, type=float,
                        help="Clustering threshold")

    parser.add_argument("-spd", "--support_decay", default=0.99999, type=float,
                        help="support factor decay")

    parser.add_argument("-tr", "--training", default=False, type=bool,
                        help="Training Good Nodes")

    # RL-Agent
    parser.add_argument("-ep", "--epochs", default=40, type=int, help="K-epochs")
    parser.add_argument("-eps", "--eps_clip", default=0.2, type=float, help="Epsilon clip")
    parser.add_argument("-ga", "--gamma", default=1, type=int, help="Gamma")
    parser.add_argument("-lra", "--lr_actor", default=0.0001, type=float, help="Learning rate actor")
    parser.add_argument("-lrc", "--lr_critic", default=0.003, type=float, help="Learning rate critic")

    args = parser.parse_args(args=[])

    file_path = '../data/' + args.dataset + '_mean_' + args.diffusion_model + str(10 * args.seed_rate) + '.SG'
    with open(file_path, 'rb') as f:
        graphs, multiplex = pickle.load(f)

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # ranking graph
    graphs = sorted(graphs, key=lambda graph: (graph.number_of_nodes(), graph.number_of_edges()))
    # node attribute
    for node in multiplex.nodes():
        multiplex.nodes[node]['attribute'] = 1.0
    budget = int(multiplex.number_of_nodes() * args.seed_rate * 0.01)

    start_time = time.time()

    chosen, costs, profits = board_generator(deepcopy(multiplex), graphs, list(multiplex.nodes()), l=budget,
                                             diffusion=args.diffusion_model)
    seed_set, budget_layers = mckp_constraint_solver(len(graphs), chosen, costs, profits, l=budget)
    budget_layers[-1] += budget - np.sum(budget_layers)
    adj_matrix = nx.to_scipy_sparse_array(deepcopy(multiplex), dtype=np.float32, format='csr')
    spread, after_activations = diffusion_evaluation(adj_matrix, seed_set, diffusion=args.diffusion_model)

    model_path = "model.pkl"
    if args.training:
        print("Training")

        training_good_nodes(deepcopy(multiplex), graphs, budget_layers, model_path=model_path)

    good_nodes = finding_good_nodes(model_path, deepcopy(multiplex))
    for graph in graphs:
        good_nodes.append(list(graph.nodes())[0])

    good_nodes.extend(seed_set)
    good_nodes = sorted(set(good_nodes))

    for node in multiplex.nodes():
        multiplex.nodes[node]['attribute'] = 1.0

    seed_set = get_seedset(deepcopy(multiplex), graphs, deepcopy(good_nodes), budget_layers, args.thresold,
                           diffusion=args.diffusion_model)
    adj_matrix = nx.to_scipy_sparse_array(deepcopy(multiplex), dtype=np.float32, format='csr')
    spread, after_activations = diffusion_evaluation(adj_matrix, seed_set, diffusion=args.diffusion_model)

    num_to_remove = len(good_nodes) // 3
    items_to_remove = random.sample(good_nodes, num_to_remove)
    good_nodes = [item for item in good_nodes if item not in items_to_remove]
    good_nodes.extend(seed_set)
    good_nodes = sorted(set(good_nodes))

    print("Len Good Nodes: ", len(good_nodes))
    # Chuyển đổi set thành dictionary
    good_nodes = {i: item for i, item in enumerate(good_nodes)}

    ####### initialize environment hyperparameters ######
    env_name = "MIM_Reasoner"
    env = GraphDreamerEnv(multiplex, budget, seed_set, good_nodes, args.support_decay, spread, 1)
    max_ep_len = budget  # max timesteps in one episode
    max_training_timesteps = int(
        len(multiplex.nodes) * 500)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(2e3)  # save model frequency (in num timesteps)
    update_timestep = max_ep_len * 50  # update policy every n timesteps
    print("training environment name : " + env_name)

    random_seed = 0
    early_stopping = True
    # initialize a PPO agent
    state_dim = budget + 1
    action_dim = len(good_nodes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ppo_agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.epochs, args.eps_clip,
                    max_training_timesteps, device=device)

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    ################### checkpointing ###################

    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    #####################################################

    ################# training procedure ################

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0


    def evaluate_performance(predicted_seed):
        state, mask, done = env.reset('nothing')
        for t in range(0, max_ep_len):
            # select action with policy
            action = ppo_agent.select_action(state, time_step, copy.deepcopy(best_found_action[t - 1]), mask, 'no_duplicate')
            state, reward, done, mask, spread = env.step(action)
            # saving reward and is_terminals
            predicted_seed.append(good_nodes[action])
        best_spread_rl = spread
        return predicted_seed, best_spread_rl


    # training loop
    predicted_seed = []
    success_rate = 0
    key_good_node_list = list(good_nodes.keys())
    val_good_node_list = list(good_nodes.values())
    best_found_action = []

    for value in env.best_action:
        # print key with val 100
        position = val_good_node_list.index(value)
        best_found_action.append(key_good_node_list[position])

    while time_step <= max_training_timesteps and early_stopping == True:

        state, mask, done = env.reset('nothing')
        spread_CELF = env.maximum_reward
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state, time_step, copy.deepcopy(best_found_action[t-1]), mask, 'duplicate')
            state, reward, done, mask, spread = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                print("Update model !!")
                predicted_seed, best_spread_rl = evaluate_performance(predicted_seed)
                ppo_agent.update(predicted_seed, best_found_action)
                predicted_seed = []

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                predicted_seed, best_spread_rl = evaluate_performance(predicted_seed)
                print("Episode : {} \t Timestep : {} \t Average Reward : {} \t Current Spread : {}"
                      " \t Current Predicted Seed : {} \t"
                      " \t RL Best Found Seed : {} \t "
                      " Support_factor : {}".format(i_episode, time_step, print_avg_reward, best_spread_rl,
                                                    predicted_seed,
                                                    env.best_action,
                                                    round(env.support_factor, 2)))

                if predicted_seed == best_found_action:
                    if early_stopping:
                        print("Early Stopping!")
                        early_stopping = False
                    env.support_factor -= 1
                    env.support_factor = max(env.support_factor, 0)

                predicted_seed = []
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                # print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

        if env.support_factor == 2:
            break

    log_f.close()

    ################################### Store results to file ###################################
    save_time = time.time() - start_time

    adj_matrix = nx.to_scipy_sparse_array(deepcopy(multiplex), dtype=np.float32, format='csr')
    # spread, after_activations = diffusion_evaluation(adj_matrix, predicted_seed, diffusion=args.diffusion_model)
    predicted_seed, spread = evaluate_performance(predicted_seed)

    print("Time", save_time)
    print("Seed set", seed_set)
    print("Budget", budget)
    print("Spread", spread)

    output_path = os.path.join('..', 'Output', file_name + '.csv')

    if not os.path.exists(output_path):
        data = [
            ["algorithm", "Nodes", "Edges", "Layer", "Budget", "Diffusion", "Seed set", "Spread", "Tỉme"],
            ["RL_MIM", len(multiplex.nodes), len(multiplex.edges), len(graphs), args.seed_rate, args.diffusion_model,
             seed_set, spread,
             save_time]
        ]
    else:
        data = [
            ["RL_MIM", len(multiplex.nodes), len(multiplex.edges), len(graphs), args.seed_rate, args.diffusion_model,
             seed_set, spread,
             save_time]
        ]

    # Save file CSV
    with open(output_path, 'a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)