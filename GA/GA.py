import random

class GeneticAlgorithm:
    def __init__(self, weight, profit, weight_limit, popsize=20, pc=0.8, pm=0.2, N=30):
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

        self.best_values = []

    def sum_pw(self, individual, profit_or_weight):
        total = 0
        for i in range(len(individual)):
            total += individual[i] * profit_or_weight[i]
        return total

    def put_out(self, individual):
        for i in range(len(individual)):
            individual[i] *= random.randint(0, 1)
        return individual

    def init_population(self):
        self.population = []
        for i in range(self.popsize):
            individual = []
            for j in range(len(self.weight)):
                individual.append(random.randint(0, 1))
            total_weight = self.sum_pw(individual, self.weight)
            while total_weight > self.weight_limit:
                individual = self.put_out(individual)
                total_weight = self.sum_pw(individual, self.weight)
            self.population.append(individual)

    def compute_fitness(self):
        total_weight = []
        total_profit = []
        for individual in self.population:
            total_weight.append(self.sum_pw(individual, self.weight))
            total_profit.append(self.sum_pw(individual, self.profit))
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

    def roulettewheel(self, total_profit):
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
        i = 0
        new_population = self.population[:]
        while i < len(self.population):
            if random.uniform(0, 1) < self.pc:
                mother_index = random.randint(0, len(self.population) - 1)
                father_index = random.randint(0, len(self.population) - 1)
                cpoint = random.randint(0, len(self.population[0]) - 1)
                if father_index != mother_index:
                    temp11 = self.population[father_index][:cpoint]
                    temp12 = self.population[father_index][cpoint:]

                    temp21 = self.population[mother_index][cpoint:]
                    temp22 = self.population[mother_index][:cpoint]

                    child1 = temp21 + temp11
                    child2 = temp12 + temp22

                    new_population[father_index] = child1
                    new_population[mother_index] = child2
            i += 1
        self.population = new_population

    def mutation(self):
        temporary = []
        for i in range(len(self.population)):
            p = random.uniform(0, 1)
            if p < self.pm:
                j = 0
                while j < 2:
                    mpoint = random.randint(0, len(self.population[0]) - 1)
                    if self.population[i][mpoint] == 0:
                        self.population[i][mpoint] = 1
                    else:
                        self.population[i][mpoint] = 0
                    j += 1
                temporary.append(self.population[i])
            else:
                temporary.append(self.population[i])
        self.population = temporary

    def solve(self):
        self.init_population()
        iter = 0
        while iter < self.N:
            iter += 1
            # print("——————————————————————————————————————————————————————————————————————————————————————————————————————")
            # print(f'第{iter}代')
            # print(f'第{iter}代群体为:', self.population)

            total_weight, total_profit = self.compute_fitness()
            # print('weight为:', total_weight)
            # print('profit为:', total_profit)

            total_weight, total_profit = self.select(total_weight, total_profit)
            # print(f'筛选后的群种为：{self.population}')
            # print(f'筛选后的weight为：{total_weight}')
            # print(f'筛选后的profit为：{total_profit}')

            self.roulettewheel(total_profit)
            # print('选择后的种群为:', self.population)

            self.crossover()
            # print('交叉后的群体为:', self.population)

            self.mutation()
            # print('变异后的群体为:', self.population)

            # print('-------------------------------' * 2)

            total_weight, total_profit = self.compute_fitness()
            total_weight, total_profit = self.select(total_weight, total_profit)
            m = max(range(len(self.population)), key=lambda x: total_profit[x])
            if total_profit[m] > self.best_fitness:
                self.best_individual = self.population[m]
                self.best_fitness = total_profit[m]
                self.best_individual_pop = self.population
                self.best_fitness_pop = total_profit
                self.best_weight = total_weight[m]
                self.best_weight_pop = total_weight


            # print("全局最优个体价值为:", self.best_fitness)
            # print("全局最优个体重量为：", self.best_weight)
            self.best_values.append(self.best_fitness)

        return self.best_values

        # print("全局最优个体种群为：", self.best_individual_pop)
        # print("全局最优个体为：", self.best_individual)
        # print("全局最优个体种群价值为:", self.best_fitness_pop)
        # print("全局最优个体价值为:", self.best_fitness)
        # print("全局最优个体重量为：", self.best_weight)
        # print("全局最优个体种群重量为：", self.best_weight_pop)