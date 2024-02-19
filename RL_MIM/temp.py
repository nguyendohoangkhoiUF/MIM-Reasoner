# import time
# import numpy as np
# import threading
# from tqdm import tqdm
#
#
# def IC(combined_graph, T, mc=100):
#     """
#         The IC() function describing the spread process
#         Input:
#             combined_graph: multiplex network
#             T: set of seed nodes
#             mc: the number of Monte-Carlo simulations
#         Output:
#             np.mean(spread): average number of nodes influenced by the seed nodes
#             np.array(after_activations): set node status
#             node_actives: set of active nodes
#     """
#
#     spread = []
#     after_activations = []
#     list_node_actives = []
#
#     for i in range(mc):
#         node_actives = set(T[:])  # initialized as a set containing the initial seed nodes T.
#         after_activation = [0] * len(combined_graph.nodes)
#         # Simulate propagation process
#         new_active, A = T[:], T[:]
#
#         while new_active:  # Enter a while loop that continues until no new nodes are activated in an iteration.
#
#             # For each newly active node, find its neighbors that become activated
#             new_ones = []
#             for node in new_active:
#                 # Determine neighbors that become infected
#                 neighboring_node = [n for n in combined_graph.neighbors(node)]
#                 neighboring_node = list(set(neighboring_node) - node_actives)
#                 success = [False] * len(neighboring_node)
#
#                 for j, node_nei in enumerate(neighboring_node):
#                     p = combined_graph.get_edge_data(node, node_nei)['weight']
#                     success[j] = np.random.uniform(0, 1) < p
#
#                 new_ones += list(np.extract(success, neighboring_node))
#                 node_actives.update(new_ones)
#             # Update the set of newly activated nodes
#             new_active = list(set(new_ones) - set(A))
#
#             # Add newly activated nodes to the set of activated nodes
#             # print(new_active)
#             A += new_active
#             for activated_node in A:
#                 after_activation[activated_node] = 1
#
#         # Update the activation status of the seed nodes
#         for i, s in enumerate(T):
#             after_activation[s] = 2
#
#         spread.append(len(A))
#         after_activations.append(after_activation)
#
#     return np.mean(spread), np.array(after_activations), list_node_actives
#
# def Evaluate(combined_graph, T, mc=100):
#     """
#         The IC() function describing the spread process
#         Input:
#             combined_graph: multiplex network
#             T: set of seed nodes
#             mc: the number of Monte-Carlo simulations
#         Output:
#             np.mean(spread): average number of nodes influenced by the seed nodes
#             np.array(after_activations): set node status
#             node_actives: set of active nodes
#     """
#
#     spread = []
#     after_activations = []
#     list_node_actives = []
#
#     for i in range(mc):
#         node_actives = set(T[:])  # initialized as a set containing the initial seed nodes T.
#         after_activation = [0] * len(combined_graph.nodes)
#         # Simulate propagation process
#         new_active, A = T[:], T[:]
#
#         while new_active:  # Enter a while loop that continues until no new nodes are activated in an iteration.
#
#             # For each newly active node, find its neighbors that become activated
#             new_ones = []
#             for node in new_active:
#                 # Determine neighbors that become infected
#                 neighboring_node = [n for n in combined_graph.neighbors(node)]
#                 neighboring_node = list(set(neighboring_node) - node_actives)
#                 success = [False] * len(neighboring_node)
#
#                 for j, node_nei in enumerate(neighboring_node):
#                     p = combined_graph.get_edge_data(node, node_nei)['weight']
#                     success[j] = np.random.uniform(0, 1) < p
#
#                 new_ones += list(np.extract(success, neighboring_node))
#                 node_actives.update(new_ones)
#             # Update the set of newly activated nodes
#             new_active = list(set(new_ones) - set(A))
#
#             # Add newly activated nodes to the set of activated nodes
#             # print(new_active)
#             A += new_active
#             for activated_node in A:
#                 after_activation[activated_node] = 1
#
#         # Update the activation status of the seed nodes
#         for i, s in enumerate(T):
#             after_activation[s] = 2
#         spread.append(len(A))
#
#
#         after_activations.append(after_activation)
#
#     return np.mean(spread), np.array(after_activations), list_node_actives
#
# def celf(combined_graph, g_i, k, mc=1000):
#     """
#         The CELF() function describes the process of finding a seed set
#         Input:
#             combined_graph: multiplex network
#             g_i: graph of the ith layer
#             k: budget amount
#             mc: the number of Monte-Carlo simulations
#         Output:
#             S: optimal seed set
#             SPREAD: resulting spread
#             timelapse: time for each iteration
#             LOOKUPS: number of spread calculations
#             data: Collection of node status
#
#     """
#
#     # --------------------
#     # Find the first node with greedy algorithm
#     # --------------------
#
#     # Calculate the first iteration sorted list
#
#     start_time = time.time()
#     marg_gain = []
#     data = None
#     degrees = dict(combined_graph.degree())
#     avg_degree = sum(degrees.values()) / len(degrees)
#
#     for node in tqdm(g_i.nodes):
#         degree = combined_graph.degree(node)
#         if degree < avg_degree:
#             break
#         s, at, _ = IC(combined_graph, [node], mc)
#         marg_gain.append(s)
#         # data = at
#         if data is None:
#             data = at
#         else:
#             data = np.concatenate((data, at), axis=0)
#
#     # Create the sorted list of nodes and their marginal gain
#     Q = sorted(zip(g_i.nodes, marg_gain), key=lambda x: x[1], reverse=True)
#     print(len(Q), len(g_i.nodes))
#
#     # Select the first node and remove from candidate list
#     S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
#
#     Q, LOOKUPS, timelapse = Q[1:], [len(g_i.nodes)], [time.time() - start_time]
#
#     # --------------------
#     # Find the next k-1 nodes using the list-sorting procedure
#     # --------------------
#
#     for _ in tqdm(range(k - 1)):
#
#         check, node_lookup = False, 0
#
#         while not check:
#             # Count the number of times the spread is computed
#             node_lookup += 1
#
#             # Recalculate spread of top node
#             current = Q[0][0]
#
#             # Evaluate the spread function and store the marginal gain in the list
#             s, at, _ = IC(combined_graph, S + [current], mc)
#             data = np.concatenate((data, at), axis=0)
#             Q[0] = (current, s - spread)
#
#             # Re-sort the list
#             Q = sorted(Q, key=lambda x: x[1], reverse=True)
#
#             # Check if previous top node stayed on top after the sort
#             check = (Q[0][0] == current)
#
#         # Select the next node
#         spread += Q[0][1]
#         S.append(Q[0][0])
#         SPREAD.append(spread)
#         LOOKUPS.append(node_lookup)
#         timelapse.append(time.time() - start_time)
#
#         # Remove the selected node from the list
#         Q = Q[1:]
#
#     return S, SPREAD, timelapse, LOOKUPS, data
#
#
#


# import speech_recognition as sr
#
#
# def audio_to_text(audio_file):
#     r = sr.Recognizer()
#
#     # Mở file audio
#     with sr.AudioFile(audio_file) as source:
#         audio = r.record(source)  # Đọc audio từ file
#
#     try:
#         # Nhận dạng giọng nói
#         text = r.recognize_google(audio, language='vi-VN')
#         return text
#     except sr.UnknownValueError:
#         print("Không thể nhận dạng giọng nói")
#     except sr.RequestError as e:
#         print(f"Lỗi trong quá trình gửi yêu cầu đến Speech Recognition service; {e}")
#
#     return ""
#
#
# print(audio_to_text(r"F:\Anh Nguyen\Reinforcement Learning\MIM_Reasoner\CTKV 1\FILE CTKV1\1104 , 03.08.2020 , 09h03 , 766666699.wav"))


