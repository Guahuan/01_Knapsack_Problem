import random
import math
from re import S
import numpy as np

class SimulatedAnnealing:
    def __init__(self, values, weights, max_weight, n_iters, max_temp, min_temp, cooling_rate):
        self.values = values
        self.weights = weights
        self.max_weight = max_weight
        self.n_iters = n_iters
        self.n_items = len(values)
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate

    def solve(self):
        solution = self.generate_valid_solution()

        self.best_solution = solution
        self.best_value = self.calculate_value(solution)
        self.best_weight = self.calculate_weight(solution)

        temp = self.max_temp
        while temp > self.min_temp:
            for iter in range(self.n_iters):
                i = random.randint(0, len(self.values) - 1)
                new_solution = solution[:]
                new_solution[i] = 1 - new_solution[i]

                new_weight = self.calculate_weight(new_solution)
                old_value = self.calculate_value(solution)
                new_value = self.calculate_value(new_solution)

                if (new_value > old_value or random.random() < math.exp((new_value - old_value) / temp)) and new_weight <= self.max_weight:
                    solution = new_solution

                if self.calculate_value(solution) > self.best_value:
                    self.best_solution = solution
                    self.best_value = self.calculate_value(solution)
                    self.best_weight = self.calculate_weight(solution)

                print("iter: ", iter)
                print("best value: ", self.best_value)
                print("best weight: ", self.best_weight)

            temp *= self.cooling_rate

    def generate_valid_solution(self):
        solution = [random.randint(0, 1) for _ in range(len(self.values))]
        while self.calculate_weight(solution) > self.max_weight:
            solution = [random.randint(0, 1) for _ in range(len(self.values))]
        return solution

    def calculate_value(self, solution):
        return sum(self.values * solution)


    def calculate_weight(self, solution):
        return sum(self.weights * solution)