import numpy as np

"""
    Refer: https://www.doc88.com/p-487729432921.html
    1、选择概率函数
    2、如何限制重量
    3、更新信息素
"""
class AntColonyOptimization:
    def __init__(self, values, weights, max_weight, costs, max_cost, n_iters, n_ants, alpha, beta, decay):
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
        self.costs = costs
        self.max_cost = max_cost
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

        self.n_item = len(values)
        self.pheromone = np.ones(self.n_item)

        self.best_value = 0
        self.best_weight = 0
        self.best_cost = 0
        self.best_solution = None

        self.solutions = []

    def solve(self):
        for iter in range(self.n_iters):
            solutions = []
            total_values = []

            # TODO: parallel
            for ant in range(self.n_ants):
                solution, total_value = self.construct_solution()
                solutions.append(solution)
                total_values.append(total_value)

            self.update_pheromone(solutions, total_values)

            self.solutions.append(total_value)

            # print("iter: ", iter)
            # print("best value: ", self.best_value)
            # print("best weight: ", self.best_weight)
            # print("best cost: ", self.best_cost)
            # # print("best solution: ", self.best_solution)

        return self.solutions


    def construct_solution(self):
        tabu = [0 for _ in range(self.n_item)]
        solution = [0 for _ in range(self.n_item)]
        total_weight = 0
        total_cost = 0
        total_value = 0

        while(0 in tabu):
            p = self.cal_probability(tabu)
            for i in range(self.n_item):
                if np.random.rand() < p[i]:
                    # ant select item i
                    tabu[i] = 1
                    if total_weight + self.weights[i] <= self.max_weight and total_cost + self.costs[i] <= self.max_cost:
                        # put in bag
                        solution[i] = 1
                        total_weight += self.weights[i]
                        total_cost += self.costs[i]
                        total_value += self.values[i]
                    break

        if total_value > self.best_value:
            self.best_value = total_value
            self.best_weight = total_weight
            self.best_cost = total_cost
            self.best_solution = solution

        return solution, total_value

    def cal_probability(self, tabu):
        tabu = np.array(tabu)
        mask = (tabu == 0)
        values = np.array(self.values)
        weights = np.array(self.weights)
        costs = np.array(self.costs)
        down = np.sum(self.pheromone[mask] ** self.alpha * (((values / weights + values / costs) * 0.5)[mask] ** self.beta))
        p = np.zeros(self.n_item)
        p[mask] = (self.pheromone[mask] ** self.alpha * (((values / weights + values / costs) * 0.5)[mask]) ** self.beta) / down
        return p.tolist()

    def update_pheromone(self, solutions, total_values):
        self.pheromone = (1 - self.decay) * self.pheromone
        solutions = np.array(solutions)
        total_values = np.array(total_values)
        self.pheromone += np.sum(solutions * self.values * total_values[:, None], axis=0)

