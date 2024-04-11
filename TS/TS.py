import pandas as pd
import random

class TabuKnapsackSolver:
    def __init__(self, values, weights, max_weight, tabu_size, max_iterations):
        self.values = values
        self.weights = weights
        self.max_weight = max_weight
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
        self.n_items = len(values)
        self.tabu_list = []

    def solve(self):
        current_solution = self.generate_initial_solution()
        best_solution = current_solution[:]
        best_value = self.calculate_value(current_solution)

        iteration = 0
        while iteration < self.max_iterations:
            neighbor_solution = self.generate_neighbor(current_solution)
            neighbor_value = self.calculate_value(neighbor_solution)

            if neighbor_value > best_value:
                best_solution = neighbor_solution[:]
                best_value = neighbor_value

            current_solution = neighbor_solution[:]

            self.update_tabu_list(current_solution)

            iteration += 1

        return best_value, self.calculate_weight(best_solution), best_solution

    def generate_initial_solution(self):
        solution = [random.choice([0, 1]) for _ in range(self.n_items)]
        while self.calculate_weight(solution) > self.max_weight:
            solution = [random.choice([0, 1]) for _ in range(self.n_items)]
        return solution

    def generate_neighbor(self, solution):
        neighbor = solution[:]
        idx = random.randint(0, self.n_items - 1)
        neighbor[idx] = 1 - neighbor[idx]  # Flip the bit
        while self.calculate_weight(neighbor) > self.max_weight or neighbor in self.tabu_list:
            idx = random.randint(0, self.n_items - 1)
            neighbor[idx] = 1 - neighbor[idx]  # Flip the bit
        return neighbor

    def update_tabu_list(self, solution):
        self.tabu_list.append(solution)
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)

    def calculate_value(self, solution):
        return sum(value * solution[i] for i, value in enumerate(self.values))

    def calculate_weight(self, solution):
        return sum(weight * solution[i] for i, weight in enumerate(self.weights))


# Data for the knapsack problem
#weights = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
#values = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
#max_weight = 269
#tabu_size = 10
#max_iterations = 100

# Unified param
diamonds = pd.read_csv('diamonds.csv')
diamonds = diamonds.iloc[:100]                              # 近似最优解: value = 32615, weight = 10
weights = diamonds['carat'].values
values = diamonds['price'].values
max_weight = 10
tabu_size = 100
max_iterations = 100

# Solve the knapsack problem using Tabu Search algorithm
tabu_solver = TabuKnapsackSolver(values, weights, max_weight, tabu_size, max_iterations)
best_value, best_weight, best_solution = tabu_solver.solve()

# Print the results
print("Best value by using TS:", best_value)
print("Best weight by using TS:", best_weight)
print("Best solution by using TS:", best_solution)
