import pandas as pd

class GreedyKnapsackSolver:
    def __init__(self, values, weights, max_weight):
        self.values = values
        self.weights = weights
        self.max_weight = max_weight
        self.n_items = len(values)
        self.selected_items = []

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

        total_value = sum(self.values[i] for i in self.selected_items)
        total_weight = sum(self.weights[i] for i in self.selected_items)

        return total_value, total_weight, self.selected_items


# Data for the knapsack problem
#weights = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
#values = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
#max_weight = 269

# Unified param
diamonds = pd.read_csv('diamonds.csv')
diamonds = diamonds.iloc[:100]                              # 近似最优解: value = 32615, weight = 10
weights = diamonds['carat'].values
values = diamonds['price'].values
max_weight = 10
n_iters = 1000


# Solve the knapsack problem using greedy algorithm
greedy_solver = GreedyKnapsackSolver(values, weights, max_weight)
total_value, total_weight, selected_items = greedy_solver.solve()

# Print the results
print("Total value by using GA:", total_value)
print("Total weight by using GA:", total_weight)
print("Selected items by using GA:", selected_items)