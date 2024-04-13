class GreedyKnapsackSolver:
    def __init__(self, values, weights, max_weight):
        self.values = values
        self.weights = weights
        self.max_weight = max_weight
        self.n_items = len(values)
        self.selected_items = []
        self.total_value = 0
        self.total_weight = 0

    def solve(self):
        remaining_weight = self.max_weight
        # Create a list of (index, value-to-weight ratio) tuples
        value_weight_ratio = [(i, self.values[i] / self.weights[i]) for i in range(self.n_items)]
        # Sort items by value-to-weight ratio in descending order
        sorted_items = sorted(value_weight_ratio, key=lambda x: x[1], reverse=True)

        for item_idx, _ in sorted_items:
            if remaining_weight >= self.weights[item_idx]:
                # If the item can fit in the knapsack, select it
                self.selected_items.append(item_idx)
                remaining_weight -= self.weights[item_idx]

        self.total_value = sum(self.values[i] for i in self.selected_items)
        self.total_weight = sum(self.weights[i] for i in self.selected_items)

        print("best value: ", self.total_value)
        print("best weight: ", self.total_weight)
