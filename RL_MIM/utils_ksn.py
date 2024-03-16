# # import numpy as np
# # import networkx as nx
# # import ndlib.models.ModelConfig as mc
# # import ndlib.models.epidemics as ep
# # import time
# # import random
# # from collections import Counter
# # from tqdm import tqdm
# # import pandas as pd


# # def diffusion_evaluation(adj_matrix, seed, diffusion='LT'):

# #     total_infect = 0
# #     G = nx.from_scipy_sparse_array(adj_matrix)
    
# #     for i in range(10):
        
# #         if diffusion == 'LT':
# #             model = ep.ThresholdModel(G)
# #             config = mc.Configuration()
# #             for n in G.nodes():
# #                 config.add_node_configuration("threshold", n, 0.5)
# #         elif diffusion == 'IC':
# #             model = ep.IndependentCascadesModel(G)
# #             config = mc.Configuration()
# #             for e in G.edges():
# #                 config.add_edge_configuration("threshold", e, 1/nx.degree(G)[e[1]])
# #         elif diffusion == 'SIS':
# #             model = ep.SISModel(G)
# #             config = mc.Configuration()
# #             config.add_model_parameter('beta', 0.001)
# #             config.add_model_parameter('lambda', 0.001)
# #         else:
# #             raise ValueError('Only IC, LT and SIS are supported.')

# #         config.add_model_initial_configuration("Infected", seed)

# #         model.set_initial_status(config)

# #         iterations = model.iteration_bunch(100)

# #         node_status = iterations[0]['status']

# #         seed_vec = np.array(list(node_status.values()))

# #         for j in range(1, len(iterations)):
# #             node_status.update(iterations[j]['status'])


# #         inf_vec = np.array(list(node_status.values()))
# #         inf_vec[inf_vec == 2] = 1

# #         total_infect += inf_vec.sum()
    
# #     return total_infect/10, list(node_status.values())


# # def get_RRS(df):   
# #     """
# #     Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
# #             p:  Disease propagation probability
# #     Return: A random reverse reachable set expressed as a list of nodes
# #     """

# #     # Step 1. Select random source node
# #     source = random.choice(np.unique(df['source']))
    
# #     # Step 2. Get an instance of g from G by sampling edges  
# #     g = df.copy().loc[np.random.uniform(0,1,df.shape[0]) < df['prob']]

# #     # Step 3. Construct reverse reachable set of the random source node
# #     new_nodes, RRS0 = [source], [source]   
# #     while new_nodes:
        
# #         # Limit to edges that flow into the source node
# #         temp = g.loc[g['target'].isin(new_nodes)]

# #         # Extract the nodes flowing into the source node
# #         temp = temp['source'].tolist()

# #         # Add new set of in-neighbors to the RRS
# #         RRS = list(set(RRS0 + temp))

# #         # Find what new nodes were added
# #         new_nodes = list(set(RRS) - set(RRS0))

# #         # Reset loop variables
# #         RRS0 = RRS[:]

# #     return(RRS)


# # def ris(G,k,p=0.5,mc=1000):    
# #     """
# #     Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
# #             k:  Size of seed set
# #             p:  Disease propagation probability
# #             mc: Number of RRSs to generate
# #     Return: A seed set of nodes as an approximate solution to the IM problem
# #     """
# #     source = []
# #     target = []
# #     prob = []
    
# #     for edges in G.edges():
# #         if G.is_directed():
# #             source.append(edges[0])
# #             target.append(edges[1]) 
# #             prob.append(G.nodes[edges[1]]['attribute'])
# #         else:
# #             source.extend(list(edges))
# #             target.extend(list(edges)) 
# #             prob.append([G.nodes[edges[1]]['attribute'], G.nodes[edges[1]]['attribute']])
    
# #     df = pd.DataFrame({'source':source,'target':target, 'prob': prob})
    
    
# #     # Step 1. Generate the collection of random RRSs
# #     start_time = time.time()
# #     R = [get_RRS(df) for _ in range(mc)]

# #     # Step 2. Choose nodes that appear most often (maximum coverage greedy algorithm)
# #     SEED, timelapse = [], []
# #     for _ in range(k):
        
# #         # Find node that occurs most often in R and add to seed set
# #         flat_list = [item for sublist in R for item in sublist]
# #         seed = Counter(flat_list).most_common()[0][0]
# #         SEED.append(seed)
        
# #         # Remove RRSs containing last chosen seed 
# #         R = [rrs for rrs in R if seed not in rrs]
        
# #         # Record Time
# #         timelapse.append(time.time() - start_time)
    
# #     return(sorted(SEED),timelapse)




# import numpy as np
# import networkx as nx
# import ndlib.models.ModelConfig as mc
# import ndlib.models.epidemics as ep
# import time
# from tqdm import tqdm


# def diffusion_evaluation(adj_matrix, seed, diffusion='LT'):

#     total_infect = 0
#     G = nx.from_scipy_sparse_array(adj_matrix)
    
#     for i in range(10):
        
#         if diffusion == 'LT':
#             model = ep.ThresholdModel(G)
#             config = mc.Configuration()
#             for n in G.nodes():
#                 config.add_node_configuration("threshold", n, 0.5)
#         elif diffusion == 'IC':
#             model = ep.IndependentCascadesModel(G)
#             config = mc.Configuration()
#             for e in G.edges():
#                 config.add_edge_configuration("threshold", e, 1/nx.degree(G)[e[1]])
#         elif diffusion == 'SIS':
#             model = ep.SISModel(G)
#             config = mc.Configuration()
#             config.add_model_parameter('beta', 0.001)
#             config.add_model_parameter('lambda', 0.001)
#         else:
#             raise ValueError('Only IC, LT and SIS are supported.')

#         config.add_model_initial_configuration("Infected", seed)

#         model.set_initial_status(config)

#         iterations = model.iteration_bunch(100)

#         node_status = iterations[0]['status']

#         seed_vec = np.array(list(node_status.values()))

#         for j in range(1, len(iterations)):
#             node_status.update(iterations[j]['status'])


#         inf_vec = np.array(list(node_status.values()))
#         inf_vec[inf_vec == 2] = 1

#         total_infect += inf_vec.sum()
    
#     return total_infect/10, list(node_status.values())



# def celf(combined_graph, graph, k, diffusion='LT'):
#     """
#         The CELF() function describes the process of finding a seed set
#         Input:
#             graph: graph network
#             k: budget amount
#             mc: the number of Monte-Carlo simulations
#         Output:
#             S: optimal seed set
#             SPREAD: resulting spread
#             timelapse: time for each iteration
#             LOOKUPS: number of spread calculations
#             Real_Dataset: Collection of node status

#     """

#     # --------------------
#     # Find the first node with greedy algorithm
#     # --------------------

#     # Calculate the first iteration sorted list

#     start_time = time.time()
#     marg_gain = []
#     Real_Dataset = None

#     for node in tqdm(graph.nodes()):
#         adj_matrix = nx.to_scipy_sparse_array(combined_graph, dtype=np.float32, format='csr')
#         infect, node_status = diffusion_evaluation(adj_matrix, [node], diffusion)
#         s = sum(combined_graph.nodes[node]['attribute'] for node in node_status if node > 0)            
#         marg_gain.append(s)

#     # Create the sorted list of nodes and their marginal gain
#     Q = sorted(zip(graph.nodes, marg_gain), key=lambda x: x[1], reverse=True)

#     # Select the first node and remove from candidate list
#     S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
#     Q, LOOKUPS, timelapse = Q[1:], [len(graph.nodes)], [time.time() - start_time]

#     # --------------------
#     # Find the next k-1 nodes using the list-sorting procedure
#     # --------------------

#     for _ in tqdm(range(k - 1)):

#         check, node_lookup = False, 0

#         if len(Q) > 0:

#             while not check:
#                 # Count the number of times the spread is computed
#                 node_lookup += 1

#                 # Recalculate spread of top node
#                 current = Q[0][0]

#                 # Evaluate the spread function and store the marginal gain in the list
#                 adj_matrix = nx.to_scipy_sparse_array(combined_graph, dtype=np.float32, format='csr')
#                 infect, node_status = diffusion_evaluation(adj_matrix, [node], diffusion)
#                 s = sum(combined_graph.nodes[node]['attribute'] for node in node_status if node > 0)  
#                 Q[0] = (current, s - spread)

#                 # Re-sort the list
#                 Q = sorted(Q, key=lambda x: x[1], reverse=True)

#                 # Check if previous top node stayed on top after the sort
#                 check = (Q[0][0] == current)

#             # Select the next node
#             spread += Q[0][1]
#             S.append(Q[0][0])
#             SPREAD.append(spread)
#             LOOKUPS.append(node_lookup)
#             timelapse.append(time.time() - start_time)

#             # Remove the selected node from the list
#             Q = Q[1:]
#         else:
#             spread = -100
#             S.append(S[-1])
#             SPREAD.append(spread)
#             LOOKUPS.append(node_lookup)
#             timelapse.append(time.time() - start_time)

#     return S, SPREAD, timelapse, LOOKUPS, Real_Dataset