import random
import numpy as np
"""
    refer：https://wenku.baidu.com/view/e8bcba23b91aa8114431b90d6c85ec3a87c28bc8.html?_wkts_=1712825921866&bdQuery=粒子群算法+01背包&needWelcomeRecommand=1
        https://blog.csdn.net/qq_43808253/article/details/130588142
    1、粒子坐标，01二进制位串
    2、粒子速度
    3、粒子更新方式
"""
class ParticleSwarmOptimization:
    def __init__(self, values, weights, max_weight, n_iters, n_particles, c1, c2, Wmax, Wmin, Vmax, Vmin):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.max_weight = max_weight
        self.n_iters = n_iters
        self.n_particles = n_particles
        self.c1 = c1
        self.c2 = c2
        self.Wmax = Wmax
        self.Wmin = Wmin
        self.Vmax = Vmax
        self.Vmin = Vmin

        self.n_item = len(values)

        self.particlesX = [[random.randint(0, 1) for _ in range(self.n_item)] for _ in range(self.n_particles)]

        self.particlesV = [[random.uniform(self.Vmin, self.Vmax) for _ in range(self.n_item)] for _ in range(self.n_particles)]

        self.g_best = self.particlesX[0]
        self.g_best_fitness = self.fitness(self.g_best)

        self.p_best = self.particlesX
        self.p_best_fitness = [self.fitness(particle) for particle in self.p_best]

        self.best_values = []


    def solve(self):
        for iter in range(self.n_iters):
            for i in range(self.n_particles):
                particle = self.particlesX[i]
                fitness = self.fitness(particle)

                if fitness > self.fitness(self.p_best[i]):
                    self.p_best[i] = particle

                if fitness > self.g_best_fitness:
                    self.g_best = particle
                    self.g_best_fitness = fitness

            w = self.Wmax - (self.Wmax - self.Wmin) * iter / self.n_iters

            for i in range(self.n_particles):
                r1 = random.random()
                r2 = random.random()
                new_V = w * np.array(self.particlesV[i]) + \
                    self.c1 * r1 * (np.array(self.p_best[i]) - np.array(self.particlesX[i])) + \
                        self.c2 * r2 * (np.array(self.g_best) - np.array(self.particlesX[i]))
                self.particlesV[i] = np.clip(new_V, self.Vmin, self.Vmax)

                new_X = 1 / (1 + np.exp(-np.array(self.particlesV[i])))
                self.particlesX[i] = [1 if vx > random.random() else 0 for vx in new_X]

            self.best_values.append(self.g_best_fitness)
            # print("iter: ", iter)
            # print("best value: ", self.g_best_fitness)
            # print("best weight: ", np.sum(np.array(self.weights) * np.array(self.g_best)))

        return self.best_values

    def fitness(self, particle):
        if np.sum(self.weights * particle) > self.max_weight:
            return 0
        else:
            return np.sum(self.values * particle)