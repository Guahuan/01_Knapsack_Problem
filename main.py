import pandas as pd
from BF.BF import BruteForceKnapsackSolver
from DP.DP import dynamic_knapsack
from Greedy.Greedy import GreedyKnapsackSolver
from TS.TS import TabuKnapsackSolver
from GA.GA import GeneticAlgorithm
from SA.SA import SimulatedAnnealing
from ACO.ACO import AntColonyOptimization
from PSO.PSO import ParticleSwarmOptimization


weights = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]         # 最优解: value = 295, weight = 269
values = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
max_weight = 269


# # Unified param
# # 近似最优解: value = 32615, weight = 10
# diamonds = pd.read_csv('diamonds.csv')
# diamonds = diamonds.iloc[:100]
# weights = diamonds['carat'].values
# values = diamonds['price'].values
# max_weight = 10
# n_iters = 1000


# BF
BF = BruteForceKnapsackSolver(values, weights, max_weight)
BF_best_value = BF.solve()


# # DP
# best_value, best_combination = dynamic_knapsack(values, weights, max_weight)
# print("最优解的总价值:", best_value)
# print("最优解的总重量:", sum(weights[int(i)] for i in range(len(best_combination)) if best_combination[i] == 1))


# # Greedy param
# Greedy = GreedyKnapsackSolver(values, weights, max_weight)
# Greedy.solve()


# # TS param
# tabu_size = 100
# TS = TabuKnapsackSolver(values, weights, max_weight, n_iters, tabu_size)
# TS.solve()


# # GA param
# popsize = 300
# pc = 0.8
# pm = 0.5
# GA = GeneticAlgorithm(weights, values, max_weight, popsize, pc, pm, n_iters)
# GA.run()


# # SA param
# max_temp = 1000
# min_temp = 1
# cooling_rate = 0.99
# SA = SimulatedAnnealing(values, weights, max_weight, n_iters, max_temp, min_temp, cooling_rate)
# SA.solve()


# # ACO param
# n_ants = 50
# alpha = 0.7
# beta = 2.3
# decay = 0.9
# ACO = AntColonyOptimization(values, weights, max_weight, n_iters, n_ants, alpha, beta, decay)
# ACO.solve()


# # PSO param
# n_particles = 1000
# c1 = 1.5
# c2 = 0.8
# Wmax = 1
# Wmin = 1
# Vmax = 10
# Vmin = -10
# PSO = ParticleSwarmOptimization(values, weights, max_weight, n_iters,n_particles, c1, c2, Wmax, Wmin, Vmax, Vmin)
# PSO.solve()