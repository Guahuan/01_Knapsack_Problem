import os
import pandas as pd
import numpy as np
from BF.BF import BruteForce
from DP.DP import DynamicProgramming
from Greedy.Greedy import GreedyKnapsackSolver
from TS.TS import TabuKnapsackSolver
from SA.SA import SimulatedAnnealing
from GA.GA import GeneticAlgorithm
from ACO.ACO import AntColonyOptimization
from PSO.PSO import ParticleSwarmOptimization


if os.path.exists('res.csv') and os.stat('res.csv').st_size > 0:
    df = pd.read_csv('res.csv')
else:
    data_dict = {
        "Brute Force": np.nan,
        "Dynamic Programming": np.nan,
        "Greedy": np.nan,
        "Tabu Search": np.nan,
        "Simulated Annealing": np.nan,
        "Genetic Algorithm": np.nan,
        "Ant Colony Optimization": np.nan,
        "Particle Swarm Optimization": np.nan
        }
    df = pd.DataFrame(data_dict, index=[0])
    df = df.reindex(index=range(10000))  # 将行数设置为10000
    df.to_csv("res.csv", index=False)


# 最优解: value = 295
weights = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
values = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
max_weight = 269
n_iters = 100


# # 最优解: value = 32616
# diamonds = pd.read_csv('diamonds.csv')
# diamonds = diamonds.iloc[:100]
# weights = diamonds['carat'].values
# values = diamonds['price'].values
# max_weight = 10
# n_iters = 1000


# # BF
# BF = BruteForce(values, weights, max_weight)
# BF_best_values = [BF.solve()]

# BF_best_values = BF_best_values + [None] * (len(df) - len(BF_best_values))
# BF_best_values_series = pd.Series(BF_best_values).ffill()
# df['Brute Force'] = BF_best_values_series
# df.to_csv('res.csv', index=False)


# # DP
# DP = DynamicProgramming(values, weights, max_weight)
# DP_best_values = [DP.solve()]

# DP_best_values = DP_best_values + [None] * (len(df) - len(DP_best_values))
# DP_best_values_series = pd.Series(DP_best_values).ffill()
# df['Dynamic Programming'] = DP_best_values_series
# df.to_csv('res.csv', index=False)


# # Greedy
# Greedy = GreedyKnapsackSolver(values, weights, max_weight)
# Greedy_best_values = [Greedy.solve()]

# Greedy_best_values = Greedy_best_values + [None] * (len(df) - len(Greedy_best_values))
# Greedy_best_values_series = pd.Series(Greedy_best_values).ffill()
# df['Greedy'] = Greedy_best_values_series
# df.to_csv('res.csv', index=False)


# # TS param
# tabu_size = 100
# TS = TabuKnapsackSolver(values, weights, max_weight, n_iters, tabu_size)
# TS_best_values = TS.solve()

# TS_best_values = TS_best_values + [None] * (len(df) - len(TS_best_values))
# TS_best_values_series = pd.Series(TS_best_values).ffill()
# df['Tabu Search'] = TS_best_values_series
# df.to_csv('res.csv', index=False)


# SA param
max_temp = 1000
min_temp = 1
cooling_rate = 0.99
SA = SimulatedAnnealing(values, weights, max_weight, n_iters, max_temp, min_temp, cooling_rate)
SA_best_values = SA.solve()

SA_best_values = SA_best_values + [None] * (len(df) - len(SA_best_values))
SA_best_values_series = pd.Series(SA_best_values).ffill()
df['Simulated Annealing'] = SA_best_values_series
df.to_csv('res.csv', index=False)


# # GA param
# popsize = 300
# pc = 0.8
# pm = 0.2
# GA = GeneticAlgorithm(weights, values, max_weight, popsize, pc, pm, n_iters)
# GA_best_values = GA.solve()

# GA_best_values = GA_best_values + [None] * (len(df) - len(GA_best_values))
# GA_best_values_series = pd.Series(GA_best_values).ffill()
# df['Genetic Algorithm'] = GA_best_values_series
# df.to_csv('res.csv', index=False)


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