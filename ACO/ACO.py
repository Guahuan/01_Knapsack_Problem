import numpy as np

# Refer: https://www.doc88.com/p-487729432921.html
#
class AntColonyOptimization:
    def __init__(self, values, weights, max_weight, n_ants, n_iters, alpha, beta, decay):
        """
        Args:
        values (list): The list of values of the items.
        weights (list): The list of weights of the items.
        max_weight(float): The volume of bag.
        n_ants (int): The number of ants.
        n_iters (int): The number of iterations.
        alpha (float): The parameter for the importance of pheromone.
        beta (float): The parameter for the importance of heuristic information (e.g., value/weight ratio of items).
        decay (float): The decay rate of the pheromone.
        """
        self.values = values
        self.weights = weights
        self.max_weight = max_weight
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

        self.n_item = len(values)
        self.pheromone = np.ones(self.n_item)

        self.best_value = 0
        self.best_weight = 0
        self.best_solution = None

    def solve(self):
        for iter in range(self.n_iters):
            solutions = []
            total_values = []

            for ant in range(self.n_ants):
                solution, total_value = self.construct_solution()
                solutions.append(solution)
                total_values.append(total_value)

            self.update_pheromone(solutions, total_values)

            print("iter: ", iter)
            print("best value: ", self.best_value)
            print("best weight: ", self.best_weight)
            # print("best solution: ", self.best_solution)

    def construct_solution(self):
        tabu = [0 for _ in range(self.n_item)]
        solution = [0 for _ in range(self.n_item)]
        total_weight = 0
        total_value = 0

        while(0 in tabu):
            p = self.cal_probability(tabu)
            for i in range(self.n_item):
                if np.random.rand() < p[i]:
                    # ant select item i
                    tabu[i] = 1
                    if total_weight + self.weights[i] <= self.max_weight:
                        # put in bag
                        solution[i] = 1
                        total_weight += self.weights[i]
                        total_value += self.values[i]
                    break

        if total_value > self.best_value:
            self.best_value = total_value
            self.best_weight = total_weight
            self.best_solution = solution

        return solution, total_value

    def cal_probability(self, tabu):
        down = 0
        for s in range(self.n_item):
            if tabu[s] == 0:
                down += self.pheromone[s] ** self.alpha * (self.values[s] / self.weights[s]) ** self.beta
        p = [0 for _ in range(self.n_item)]
        for i in range(self.n_item):
            if tabu[i] == 0:
                p[i] = (self.pheromone[i] ** self.alpha * (self.values[i] / self.weights[i]) ** self.beta) / down
        return p

    def update_pheromone(self, solutions, total_values):
        self.pheromone = (1 - self.decay) * self.pheromone
        for k in range(len(solutions)):
            self.pheromone += np.array(solutions[k]) * self.values * total_values[k]

