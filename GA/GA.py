import random
import pandas as pd


class GeneticAlgorithm:
    def __init__(self, weight, profit, weight_limit, popsize=150, pc=0.8, pm=0.2, N=30):
        self.weight = weight
        self.profit = profit
        self.weight_limit = weight_limit
        self.popsize = popsize
        self.pc = pc
        self.pm = pm
        self.N = N
        self.population = []
        self.best_individual = []
        self.best_individual_pop = []
        self.best_fitness = 0
        self.best_fitness_pop = []
        self.best_weight = 0
        self.best_weight_pop = []

    '''def init_population(self, n):
        self.population = []
        for i in range(self.popsize):
            individual = [random.randint(0, 1) for _ in range(n)]
            self.population.append(individual)'''

    def init_population(self, n):
        self.population = []
        for _ in range(self.popsize):
            individual = [random.randint(0, 1) for _ in range(n)]
            weight_sum = sum([self.weight[i] for i in range(n) if individual[i] == 1])
            while weight_sum > self.weight_limit:
                individual = [random.randint(0, 1) for _ in range(n)]
                weight_sum = sum([self.weight[i] for i in range(n) if individual[i] == 1])
            self.population.append(individual)

    def compute_fitness(self):
        total_weight = []
        total_profit = []
        for individual in self.population:
            weight_sum = 0
            profit_sum = 0
            for i in range(len(individual)):
                if individual[i] == 1:
                    weight_sum += self.weight[i]
                    profit_sum += self.profit[i]
            total_weight.append(weight_sum)
            total_profit.append(profit_sum)
        return total_weight, total_profit

    def select(self, total_weight, total_profit):
        new_population = []
        w = []
        p = []
        m = 0
        for i in range(len(total_weight)):
            if total_weight[i] < self.weight_limit:
                new_population.append(self.population[i])
                w.append(total_weight[i])
                p.append(total_profit[i])
            else:
                m += 1
        while m > 0:
            i = random.randint(0, len(new_population) - 1)
            new_population.append(new_population[i])
            w.append(w[i])
            p.append(p[i])
            m -= 1
        self.population = new_population
        return w, p

    def roulette_wheel(self, total_profit):
        sum_profit = sum(total_profit)
        p = [profit / sum_profit for profit in total_profit]
        new_population = []
        while len(new_population) < self.popsize:
            select_p = random.uniform(0, 1)
            for i in range(len(p)):
                if select_p <= sum(p[:i+1]):
                    new_population.append(self.population[i])
                    break
        self.population = new_population

    def crossover(self):
        for i in range(0, len(self.population), 2):
            if random.uniform(0, 1) < self.pc:
                cpoint = random.randint(0, len(self.population[0]) - 1)
                self.population[i][:cpoint], self.population[i+1][:cpoint] = \
                    self.population[i+1][:cpoint], self.population[i][:cpoint]

    def mutation(self):
        for individual in self.population:
            if random.uniform(0, 1) < self.pm:
                for _ in range(2):
                    mpoint = random.randint(0, len(individual) - 1)
                    individual[mpoint] = 1 - individual[mpoint]

    def run(self, verbose=True):
        n = len(self.weight)
        self.init_population(n)
        iter = 0
        while iter < self.N:
            iter += 1
            if verbose:
                print("——————————————————————————————————————————————————————————————————————————————————————————————————————")
                print(f'第{iter}代')
                print(f'第{iter}代群体为:', self.population)

            total_weight, total_profit = self.compute_fitness()
            if verbose:
                print('weight为:', total_weight)
                print('profit为:', total_profit)

            total_weight, total_profit = self.select(total_weight, total_profit)
            if verbose:
                print(f'筛选后的群种为：{self.population}')
                print(f'筛选后的weight为：{total_weight}')
                print(f'筛选后的profit为：{total_profit}')

            self.roulette_wheel(total_profit)
            if verbose:
                print('选择后的种群为:', self.population)

            self.crossover()
            if verbose:
                print('交叉后的群体为:', self.population)

            self.mutation()
            if verbose:
                print('变异后的群体为:', self.population)

            if verbose:
                print('-------------------------------' * 2)
 
            total_weight, total_profit = self.compute_fitness()
            total_weight, total_profit = self.select(total_weight, total_profit)  # 筛选weight是否大于weight_limit
            m = max(range(len(self.population)), key=lambda x: total_profit[x])
            if total_profit[m] > self.best_fitness:
                self.best_individual = self.population[m]
                self.best_fitness = total_profit[m]
                self.best_individual_pop = self.population
                self.best_fitness_pop = total_profit
                self.best_weight = total_weight[m]
                self.best_weight_pop = total_weight

        if verbose:
            print("全局最优个体种群为：", self.best_individual_pop)
            print("全局最优个体为：", self.best_individual)
            print("全局最优个体种群价值为:", self.best_fitness_pop)
            print("全局最优个体价值为:", self.best_fitness)
            print("全局最优个体种群重量为：", self.best_weight_pop)
            print("全局最优个体重量为：", self.best_weight)


def main():
    diamonds = pd.read_csv("E:\\2024春夏学期\\软计算与决策\\0-1背包问题\\diamonds.csv")
    diamonds = diamonds.iloc[:100]
    weight = diamonds['carat'].values
    value = diamonds['price'].values
    weight_limit = 20
    

    ga = GeneticAlgorithm(weight, value, weight_limit)
    ga.run()


if __name__ == "__main__":
    main()
