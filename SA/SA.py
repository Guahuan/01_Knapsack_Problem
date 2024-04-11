#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import math
import numpy as np
import pandas as pd


# In[2]:


# 背包容量c, 有n个物品
# time 迭代次数, balance平衡次数
# best 记录全局最优  T 温度  rate退火率
# wsum 当前重量总和    vsum 当前价值总和
n=100
T = 200.0
rate = 0.95
time = 500
balance = 100
global c, best, wsum, vsum
# best_way 记录全局最优解方案   now_way 记录当前解方案
best_way = [0] * n     # 将[0]复制了m遍，变成[0,0,...,0]
now_way=[0] * n


# In[3]:


#读取求解的数据
#weight = [2, 3, 5, 1, 4]
#value = [2, 5, 8, 3, 6]
diamonds = pd.read_csv("diamonds.csv")
diamonds = diamonds.iloc[:100]
weight = diamonds['carat'].values
value= diamonds['price'].values
def init():
    global c, best
    c = 10
    best = -1
    produce() #生成初始解


# In[4]:


def produce():  # 用于产生新解
    while(1):
        for k in range(n):
            if(random.random() < 0.5):
                now_way[k] = 1
            else:
                now_way[k] = 0
        calc(now_way)
        if (wsum < c):
            break
        global best
        best = calc(now_way)
        cop(best_way, now_way, n)


# In[5]:


def anneal():  # 退火函数
    global best, T, balance
    test = [0] * n
    value_current = 0     # 当前背包价值
    for i in range(balance):
        value_current = calc(now_way)    # 计算当前背包的价值
        cop(test, now_way, n)  # 记录当前方案
        ob = random.randint(0, n-1)   # 随机选取某个物品
        if (test[ob] == 1):# 在背包中则将其拿出，并加入其它物品
            put(test)
            test[ob] = 1
        else:
            if(random.random()< 0.5): #接受新解的概率
                test[ob] = 1
            else:
                get(test)
                test[ob] = 1
        value_new = calc(test)
        if(wsum > c):    # 超重，则跳过
            continue
        if (value_new > value_current): # 接受该解
            cop(now_way, test, n)
        if (value_new > best):
            best = value_new
            cop(best_way, test, n) # 更新全局最优
        else:  # 按一定的概率接受该解
            g = 1.0 * (value_new - value_current) / T
            if (random.random() < math.exp(g)): # 概率接受劣解
                cop(now_way, test, n)


# In[6]:


def cop(a, b, len):   # 把 b数组的值赋值a数组
    for i in range(len):
        a[i] = b[i]
def calc(x):        # 计算背包价值
    global c, wsum
    vsum = 0
    wsum = 0
    for i in range(n):
        vsum += x[i] * value[i]
        wsum += x[i] * weight[i]
    return vsum
def put(x):     # 往背包随机放入物品
    while(1):
        ob = random.randint(0, n-1);
        if (x[ob] == 0):   # 当前情况下物品还不在包里面
            x[ob] = 1
            break
def get(x):     # 往背包随机拿出物品
    while(1):
        ob = random.randint(0, n-1);
        if (x[ob] == 1):
            x[ob] = 0
            break


# In[9]:


if __name__ == '__main__':
    init()
    flag = 0;          # 找到最优解

    for i in range(time):
        anneal()
        T = T * rate   # 温度下降
    print('找到最终解：', best, '迭代次数', time)

    print('方案为：', best_way)

    print("weight", sum(best_way * weight))



