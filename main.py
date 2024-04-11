import pandas as pd
from ACO.ACO import AntColonyOptimization
from PSO.PSO import ParticleSwarmOptimization


# weights = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]         # 最优解: value = 295, weight = 269
# values = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
# max_weight = 269


# Unified param
diamonds = pd.read_csv('diamonds.csv')
diamonds = diamonds.iloc[:100]                              # 近似最优解: value = 32615, weight = 10
weights = diamonds['carat'].values
values = diamonds['price'].values
max_weight = 10
n_iters = 1000


# ACO param
n_ants = 50
alpha = 0.7
beta = 2.3
decay = 0.9
ACO = AntColonyOptimization(values, weights, max_weight, n_iters, n_ants, alpha, beta, decay)
ACO.solve()


# PSO param
n_particles = 1000
c1 = 1.5
c2 = 0.8
Wmax = 1
Wmin = 1
Vmax = 10
Vmin = -10
PSO = ParticleSwarmOptimization(values, weights, max_weight, n_iters,n_particles, c1, c2, Wmax, Wmin, Vmax, Vmin)
PSO.solve()