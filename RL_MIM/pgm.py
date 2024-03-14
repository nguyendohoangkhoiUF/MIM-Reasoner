import pandas as pd
from pgmpy.estimators import TreeSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
import random
from correlation import get_correlation


def get_pgm(node_status, seed_set, combined_graph, thresold=0.5):
    """
        The get_pgm() calculates the influence of a seed set on a network.
        Input:
            node_status: Collection of node status
            seed_set: Seed set
            combined_graph:  multiplex network
        Output:
            combined_graph: new multiplex network
    """
    groups = get_correlation(node_status, seed_set, thresold=thresold)
    # print(groups)

    df = pd.DataFrame(node_status)
    df = df.loc[:, list(groups.keys())]
    df = df.drop(df.columns[df.eq(0).all()], axis=1)

    columns = df.columns
    # print(len(columns))
    df = df.drop_duplicates()

    est = TreeSearch(data=df)
    estimated_model = est.estimate(estimator_type="chow-liu")
    estimated_model.edges()
    model = BayesianNetwork(estimated_model.edges())
    model.fit(data=df)
    model_exact_infer = VariableElimination(model)
    variables = list(set(model_exact_infer.variables))

    node_pgm = [1 for _ in combined_graph.nodes()]
    for node in variables:
        # tasty_infer = model_exact_infer.query(variables=[node], evidence=evidence)
        tasty_infer = model_exact_infer.query(variables=[node])
        node_pgm[node] = max(0, 1 - 4 * tasty_infer.values[1])
        # print(node_pgm[node])
        for node_corr in groups[node]:
            node_pgm[node_corr] = max(0, tasty_infer.values[1])
        
    for node in seed_set:
        node_pgm[node] = 0
    return node_pgm
