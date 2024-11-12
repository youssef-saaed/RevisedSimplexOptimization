from lpp_optimize import RevisedSimplexOptimize

cost = [3, 2]
A_ub = [
    [4, -1],
    [4, 3],
    [4, 1],
]   
b_ub = [8, 12, 8]

# A_eq = [
#     [3, 1],
# ]
# b_eq = [3]

problem = RevisedSimplexOptimize(cost, A_ub, b_ub)#, A_eq, b_eq)
print(problem.solve()) 