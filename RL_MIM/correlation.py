import math
import random
import pandas as pd


def get_correlation(node_status, seed_set, thresold):
    """
        Input:
            node_status: Collection of node status
            seed_set: Seed set
            thresold:  multiplex network
        Output:
            groups: Dictionary containing the correlation results
    """

    # Initialize the correlation matrix Q
    df = pd.DataFrame(node_status)
    df = df.drop_duplicates()
    correlation_matrix = df.corr()

    #  Initialize the set of clustered nodes S
    column_names = df.columns.tolist()
    S = [node for node in column_names]

    # Initialize the set of nodes to be considered for correlation A
    A = []
    C = [node for node in column_names]

    # Initialize a set of representative nodes P
    groups = {}

    for node in df.columns[df.eq(0).all()]:
        groups[node] = [node]
        S.remove(node)

    while len(S) > 0:
        v = S[0]
        groups[v] = [v]
        S.remove(v)
        A.append(v)
        while len(A) > 0:
            u = A[0]
            A.remove(u)
            C.remove(u)
            correlation_matrix_i = correlation_matrix[u].to_dict()
            sorted_indices = dict(sorted(correlation_matrix_i.items(), key=lambda item: item[1], reverse=True))

            for indice, value in list(sorted_indices.items())[1:]:
                if (indice in S) and (correlation_matrix_i[indice] > thresold) and (indice not in seed_set):
                    S.remove(indice)
                    groups[v].append(indice)
                    if indice in C:
                        A.append(indice)

    return groups
