import itertools

class BruteForceKnapsackSolver:
    def __init__(self, values, weights, max_weight):
        self.values = values
        self.weights = weights
        self.max_weight = max_weight
        self.n_items = len(values)

    def solve(self):
        max_value = 0
        best_combination = None

        # Generate all possible combinations of items
        for r in range(1, self.n_items + 1):
            for combination in itertools.combinations(range(self.n_items), r):
                weight = sum(self.weights[i] for i in combination)
                if weight <= self.max_weight:
                    value = sum(self.values[i] for i in combination)
                    if value > max_value:
                        max_value = value
                        best_combination = combination

        print("best value: ", max_value)
        print("best weight: ", sum(self.weights[i] for i in best_combination))