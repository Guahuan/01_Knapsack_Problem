def dynamic_knapsack(values, weights, c):
    n = len(values)
    dp = [[0] * (c + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(c + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    best_value = dp[n][c]
    best_combination = [0] * n
    w = c
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            best_combination[i - 1] = 1
            w -= weights[i - 1]

    return best_value, best_combination