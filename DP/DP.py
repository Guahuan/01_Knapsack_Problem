class DynamicProgramming:
    def __init__(self, values, weights, max_weight):
        self.scale_factor = 10 ** (self._get_decimal_places(weights + [max_weight]))
        self.values = values
        self.weights = [int(w * self.scale_factor) for w in weights]
        self.max_weight = int(max_weight * self.scale_factor)
        self.n_items = len(values)
        self.dp = [[0 for _ in range(self.max_weight + 1)] for _ in range(self.n_items + 1)]

    def _get_decimal_places(self, numbers):
        return max([self._count_decimal_places(n) for n in numbers])

    def _count_decimal_places(self, number):
        str_num = str(number)
        if '.' in str_num:
            return len(str_num) - str_num.index('.') - 1
        return 0

    def solve(self):
        for i in range(1, self.n_items + 1):
            for w in range(1, self.max_weight + 1):
                if self.weights[i - 1] <= w:
                    self.dp[i][w] = max(self.dp[i - 1][w], self.values[i - 1] + self.dp[i - 1][w - self.weights[i - 1]])
                else:
                    self.dp[i][w] = self.dp[i - 1][w]

        return self.dp[self.n_items][self.max_weight]

    def get_best_combination(self):
        best_combination = []
        w = self.max_weight
        for i in range(self.n_items, 0, -1):
            if self.dp[i][w] != self.dp[i - 1][w]:
                best_combination.append(i - 1)
                w -= self.weights[i - 1]
        return best_combination