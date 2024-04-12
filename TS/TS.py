import pandas as pd
import random

class TabuKnapsackSolver:
    def __init__(self, values, weights, max_weight, n_iters, tabu_size):
        self.values = values
        self.weights = weights
        self.max_weight = max_weight
        self.n_iters = n_iters
        self.n_items = len(values)
        self.tabu_size = tabu_size
        self.tabu_list = []

    def solve(self):
        current_solution = self.generate_initial_solution()
        best_solution = current_solution[:]
        best_value = self.calculate_value(current_solution)

        for iter in range(self.n_iters):
            neighbor_solution = self.generate_neighbor(current_solution)
            neighbor_value = self.calculate_value(neighbor_solution)

            if neighbor_value > best_value:
                best_solution = neighbor_solution[:]
                best_value = neighbor_value

            current_solution = neighbor_solution[:]

            self.update_tabu_list(current_solution)

            print("iter: ", iter)
            print("best value: ", best_value)
            print("best weight: ", self.calculate_weight(best_solution))

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
        return sum(self.values * solution)

    def calculate_weight(self, solution):
        return sum(self.weights * solution)