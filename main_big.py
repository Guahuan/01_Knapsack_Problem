import os
import select
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from BF.BF import BruteForce
from DP.DP import DynamicProgramming
from Greedy.Greedy import GreedyKnapsackSolver
from TS.TS import TabuKnapsackSolver
from SA.SA import SimulatedAnnealing
from GA.GA import GeneticAlgorithm
from ACO.ACO import AntColonyOptimization
from PSO.PSO import ParticleSwarmOptimization


# set all the parameters here
output_file = './output/big_C&W.csv'
max_data_length = 1500


# 最优解: value = 32616
diamonds = pd.read_csv('./input/diamonds.csv')
diamonds = diamonds.iloc[:100]
weights = diamonds['carat'].values
max_weight = 10
values = diamonds['price'].values


if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
    df = pd.read_csv(output_file)
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
    df = df.reindex(index=range(max_data_length))
    df.to_csv(output_file, index=False)


# # BF
# BF = BruteForce(values, weights, max_weight)
# BF_best_values = [BF.solve()]
# BF_best_values = BF_best_values + [None] * (len(df) - len(BF_best_values))
# BF_best_values_series = pd.Series(BF_best_values).ffill()
# df['Brute Force'] = BF_best_values_series
# df.to_csv(output_file, index=False)
# print('BF done')


# # DP
# DP = DynamicProgramming(values, weights, max_weight)
# DP_best_values = [DP.solve()]
# DP_best_values = DP_best_values + [None] * (len(df) - len(DP_best_values))
# DP_best_values_series = pd.Series(DP_best_values).ffill()
# df['Dynamic Programming'] = DP_best_values_series
# df.to_csv(output_file, index=False)
# print('DP done')


# # Greedy
# Greedy = GreedyKnapsackSolver(values, weights, max_weight)
# Greedy_best_values = [Greedy.solve()]
# Greedy_best_values = Greedy_best_values + [None] * (len(df) - len(Greedy_best_values))
# Greedy_best_values_series = pd.Series(Greedy_best_values).ffill()
# df['Greedy'] = Greedy_best_values_series
# df.to_csv(output_file, index=False)
# print('Greedy done')


# # TS param
# n_iters = 500
# tabu_size = 200
# TS = TabuKnapsackSolver(values, weights, max_weight, n_iters, tabu_size)
# TS_best_values = TS.solve()
# TS_best_values = TS_best_values + [None] * (len(df) - len(TS_best_values))
# TS_best_values_series = pd.Series(TS_best_values).ffill()
# df['Tabu Search'] = TS_best_values_series
# df.to_csv(output_file, index=False)
# print('TS done')


# SA param
# n_iters = 1000
# max_temp = 1000
# min_temp = 1
# cooling_rate = 0.995
# SA = SimulatedAnnealing(values, weights, max_weight, n_iters, max_temp, min_temp, cooling_rate)
# SA_best_values = SA.solve()
# SA_best_values = SA_best_values + [None] * (len(df) - len(SA_best_values))
# SA_best_values_series = pd.Series(SA_best_values).ffill()
# df['Simulated Annealing'] = SA_best_values_series
# df.to_csv(output_file, index=False)
# print('SA done')


# # GA param
# n_iters = 1000
# popsize = 150
# pc = 0.8
# pm = 0.3
# GA = GeneticAlgorithm(weights, values, max_weight, popsize, pc, pm, n_iters)
# GA_best_values = GA.solve()
# GA_best_values = GA_best_values + [None] * (len(df) - len(GA_best_values))
# GA_best_values_series = pd.Series(GA_best_values).ffill()
# df['Genetic Algorithm'] = GA_best_values_series
# df.to_csv(output_file, index=False)
# print('GA done')


# # ACO param
# n_iters = 20
# n_ants = 100
# alpha = 0.7
# beta = 2.3
# decay = 0.9
# ACO = AntColonyOptimization(values, weights, max_weight, n_iters, n_ants, alpha, beta, decay)
# ACO_best_values = ACO.solve()
# ACO_best_values = ACO_best_values + [None] * (len(df) - len(ACO_best_values))
# ACO_best_values_series = pd.Series(ACO_best_values).ffill()
# df['Ant Colony Optimization'] = ACO_best_values_series
# df.to_csv(output_file, index=False)
# print('ACO done')


# # PSO param
# n_iters = 1000
# n_particles = 300
# c1 = 1.5
# c2 = 0.8
# Wmax = 1
# Wmin = 1
# Vmax = 10
# Vmin = -10
# PSO = ParticleSwarmOptimization(values, weights, max_weight, n_iters,n_particles, c1, c2, Wmax, Wmin, Vmax, Vmin)
# PSO_best_values = PSO.solve()
# PSO_best_values = PSO_best_values + [None] * (len(df) - len(PSO_best_values))
# PSO_best_values_series = pd.Series(PSO_best_values).ffill()
# df['Particle Swarm Optimization'] = PSO_best_values_series
# df.to_csv(output_file, index=False)
# print('PSO done')


plt.figure(figsize=(12, 8))  # 增大图表的尺寸
for column in df.columns:
    plt.plot(df[column], label=column, linewidth=2)  # 增大线条的宽度
plt.xlabel('Iteration', fontsize=14)  # 增大标签的字体大小
plt.ylabel('Value', fontsize=14)  # 增大标签的字体大小
plt.title('Performance of Optimization Algorithms', fontsize=16)  # 增大标题的字体大小
plt.legend(fontsize=12)  # 增大图例的字体大小
plt.show()