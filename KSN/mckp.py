from ortools.sat.python import cp_model


def mckp_constraint_solver(num_layer, chosen, costs, profits, l):
    """
        The mckp_constraint_solver() solves a multiple-choice knapsack problem (MCKP) using a constraint programming approach
        Input:
            num_layer: Number of layers
            chosen:  List of seed set for each layer and budget (nested list)
            costs: Cost values for each item in each layer and budget (nested list)
            profits: Profit values for each item in each layer and budget (nested list)
            l: Budget constraint (integer)
        Output:
            seed_set: List of chosen items from the optimal solution
            results: List of strings describing each chosen item, including its seed set, profit, and cost
    """

    model = cp_model.CpModel()
    x = {}
    n = num_layer
    for i in range(n):
        for j in range(1, l + 1):
            # whether the jth item in the ith class is chosen
            x[i, j] = model.NewIntVar(0, 1, f"x_{i}_{j}")

    # constraints
    # the total number of items chosen in each class must be 1
    for i in range(n):
        model.Add(sum(x[i, j] for j in range(1, l + 1)) == 1)

    # each item can be chosen or not
    for i in range(n):
        for j in range(1, l + 1):
            model.AddAllowedAssignments([x[i, j]], [(0,), (1,)])

    # the total cost cannot exceed the budget
    model.Add(sum(costs[i][j] * x[i, j] for j in range(1, l + 1) for i in range(n)) <= l)

    # maximizing the total profit
    obj = []
    for i in range(n):
        for j in range(1, l + 1):
            obj.append(cp_model.LinearExpr.Term(x[i, j], profits[i][j]))
    model.Maximize(cp_model.LinearExpr.Sum(obj))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    seed_set = []
    results = []
    if status == cp_model.OPTIMAL:
        # print(f"Total profit: {solver.ObjectiveValue()}")
        total_cost = 0
        for i in range(n):
            for j in range(1, l + 1):
                if solver.Value(x[i, j]) == 1:
                    seed_set.extend(chosen[i][j])
                    res = f"Seed set : {chosen[i][j]}, profit: {profits[i][j]}, cost: {costs[i][j]}"
                    results.append(res)
                    total_cost += costs[i][j]

        # print(f"Total cost: {total_cost}")

    return seed_set, results
