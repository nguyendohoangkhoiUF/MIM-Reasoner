import csv
import os
import pickle
from model import *

from agent import PPO
from board_generator import board_generator
from environment import GraphDreamerEnv
from mckp import mckp_constraint_solver
from utils_ksn import *
from warmup import get_seedset
import argparse
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KSN algorithm")
    datasets = ['Data', 'facebook-twitter']
    parser.add_argument("-d", "--dataset", default="Data", type=str,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    num_node = [600, 5000]
    parser.add_argument("-nn", "--num_node", default=600, type=int,
                        help="one of: {}".format(", ".join(str(sorted(num_node)))))
    num_layer = [3, 4, 5, 6, 7, 8, 9]
    parser.add_argument("-nl", "--num_layer", default=3, type=int,
                        help="one of: {}".format(", ".join(str(sorted(num_layer)))))
    overlaping_user = [30, 50, 70]
    parser.add_argument("-ou", "--overlaping_user", default=30, type=int,
                        help="one of: {}".format(", ".join(str(sorted(overlaping_user)))))
    budgets = [10, 20, 30]
    parser.add_argument("-b", "--budget", default=30, type=int,
                        help="one of: {}".format(", ".join(str(sorted(budgets)))))

    parser.add_argument("-m", "--mc", default=30, type=int,
                        help="the number of Monte-Carlo simulations")

    parser.add_argument("-t", "--thresold", default=0.8, type=float,
                        help="Clustering threshold")

    # RL-Agent
    parser.add_argument("-ep", "--epochs", default=40, type=int, help="K-epochs")
    parser.add_argument("-eps", "--eps_clip", default=0.2, type=float, help="Epsilon clip")
    parser.add_argument("-ga", "--gamma", default=1, type=int, help="Gamma")
    parser.add_argument("-lra", "--lr_actor", default=0.0001, type=float, help="Learning rate actor")
    parser.add_argument("-lrc", "--lr_critic", default=0.001, type=float, help="Learning rate critic")

    args = parser.parse_args(args=[])

    file_path = '../Dataset/' + args.dataset + '/graph_' + str(args.num_node) + '_node_' + str(args.num_layer) + \
                '_layer_' + str(args.overlaping_user) + '_overlaping_user.pickle'

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    graphs = data[0]
    combined_graph = data[1]

    # ranking graph
    graphs = sorted(graphs, key=lambda graph: (graph.number_of_nodes(), graph.number_of_edges()))

    start_time = time.time()
    good_nodes = []

    gat = GAT(input_size=50)
    gat.load_state_dict(torch.load("../GCN-finding-good-nodes/model.pth"))
    gat.eval()

    for graph in graphs:
        g_node = eval_node_classifier(gat, graph)
        good_nodes.append(g_node)

    g_node = eval_node_classifier(gat, combined_graph)
    good_nodes.append(g_node)

    chosen, costs, profits = board_generator(graphs, good_nodes, l=args.budget, mc=args.mc)
    budget_layers = mckp_constraint_solver(len(graphs), chosen, costs, profits, l=args.budget)
    budget_layers[-1] += args.budget - np.sum(budget_layers)

    print(budget_layers)
    seed_set = get_seedset(combined_graph, graphs, budget_layers, good_nodes, args.thresold, mc=args.mc)

    spread, after_activations, list_node_actives = IC(combined_graph, seed_set, args.mc)

    ####### initialize environment hyperparameters ######

    env_name = "MIM_Reasoner"
    env = GraphDreamerEnv(combined_graph, args.budget, seed_set, spread, 1)
    max_ep_len = args.budget  # max timesteps in one episode
    max_training_timesteps = int(
        len(combined_graph.nodes) * 25)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(2e3)  # save model frequency (in num timesteps)
    update_timestep = max_ep_len * 20  # update policy every n timesteps
    print("training environment name : " + env_name)

    random_seed = 0
    early_stopping = True
    # initialize a PPO agent
    state_dim = args.budget + 1
    action_dim = len(combined_graph.nodes)
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
        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state, 1, 1, mask, 'no_duplicate')
            state, reward, done, mask, spread = env.step(action)
            # saving reward and is_terminals
            predicted_seed.append(action)
        best_spread_rl = spread
        return predicted_seed, best_spread_rl


    # training loop
    predicted_seed = []
    success_rate = 0
    support_factor = 0.5

    while time_step <= max_training_timesteps and early_stopping == True:

        state, mask, done = env.reset('nothing')
        spread_CELF = env.maximum_reward
        current_ep_reward = 0
        best_found_seed = env.best_action

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state, time_step, copy.deepcopy(best_found_seed[t - 1]), mask, 'duplicate')
            state, reward, done, mask, spread = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                predicted_seed, best_spread_rl = evaluate_performance(predicted_seed)
                ppo_agent.update(predicted_seed, best_found_seed)
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
                print("Episode : {} \t Timestep : {} \t Average Reward : {}"
                      " \t Current Predicted Seed : {} \t"
                      " \t RL Best Found Seed : {} \t "
                      " Support_factor : {}".format(i_episode, time_step, print_avg_reward, predicted_seed,
                                                    best_found_seed,
                                                    round(env.support_factor, 2)))

                if predicted_seed == best_found_seed:
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

    output_path = os.path.join('..', 'Output', file_name + '.csv')

    if not os.path.exists(output_path):
        data = [
            ["algorithm", "Nodes", "Edges", "Layer", "Budget", "Seed set", "Spread", "Tỉme"],
            ["RL_MIM", len(combined_graph.nodes), len(combined_graph.edges), len(graphs), args.budget, seed_set, spread,
             save_time]
        ]
    else:
        data = [
            ["RL_MIM", len(combined_graph.nodes), len(combined_graph.edges), len(graphs), args.budget, seed_set, spread,
             save_time]
        ]

    # Save file CSV
    with open(output_path, 'a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)