from search import init_population, genetic_algorithm
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_random_cities(city_num=5, length=100):
    cities = []
    xs = []
    ys = []
    for i in range(city_num):
        x, y = random.uniform(0, length), random.uniform(0, length)
        cities.append((x, y))
        xs.append(x)
        ys.append(y)
    return cities


def fitness_fn(state):
    length = len(state)
    xs = np.zeros(length)
    ys = np.zeros(length)
    x2s = np.zeros(length)
    y2s = np.zeros(length)
    for i in state:
        xs[i] = cities[state[i]][0]
        ys[i] = cities[state[i]][1]

    x2s[1:] = xs[:length-1]
    x2s[0] = xs[length-1]

    y2s[1:] = ys[:-1]
    y2s[0] = ys[-1]

    value = np.exp(-np.log(np.sum(np.sqrt((x2s - xs)**2 + (y2s - ys)**2)) + 0.00001) * 5 + 35)
    return value

city_num = 15
cities = generate_random_cities(city_num=city_num, length=100)
population = init_population(pop_number=1000, gene_pool=None, state_length=city_num)

path, history = genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=200, pmut=0.1)

print(path)

xs = []
ys = []
for i in range(len(path)):
    xs.append(cities[path[i]][0])
    ys.append(cities[path[i]][1])
xs.append(cities[path[0]][0])
ys.append(cities[path[0]][1])

plt.figure(1)
plt.plot(xs, ys)
plt.axis([-1, 100. + 1, -1, 100 + 1])

plt.figure(2)
x = []
for i in range(len(history)):
    x.append(i * 10)

plt.plot(x, history)
plt.ylabel('evaluation function')
plt.xlabel('evolution period')
plt.show()

