from search import Problem, hill_climbing
import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


class TSPHillClimbingProblem(Problem):
    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""
    def __init__(self, initial, goal=None, cities=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.cities = cities
        self.nodes_cnt = 0
        self.city_num = len(cities)
        self.no_improve_cnt = 0
        self.best_value = 0
        Problem.__init__(self, initial, goal)

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        actions = []
        state_list = list(state)
        pos = random.sample(range(1, self.city_num-1), 2)
        pos.sort()
        path1 = state_list[0:pos[0]]
        path2 = state_list[pos[0]:pos[1]]
        path3 = state_list[pos[1]:]

        actions.append(tuple(path1 + path2 + path3))
        actions.append(tuple(path1 + path3 + path2))
        actions.append(tuple(path2 + path1 + path3))
        actions.append(tuple(path2 + path3 + path1))
        actions.append(tuple(path3 + path1 + path2))
        actions.append(tuple(path3 + path2 + path1))

        return actions

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        return action

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        return self.no_improve_cnt > 50000

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        xs = []
        ys = []
        for i in state:
            xs.append(self.cities[state[i]][0])
            ys.append(self.cities[state[i]][1])
        xs_np = np.array(xs)
        ys_np = np.array(ys)

        x2s = xs_np
        x2s[1:] = xs_np[0:-1]
        x2s[0] = xs_np[-1]

        y2s = ys_np
        y2s[1:] = ys_np[0:-1]
        y2s[0] = ys_np[-1]

        value = 1 / np.sum(np.sqrt((x2s - xs)**2 + (y2s - ys)**2))

        if value > self.best_value:
            self.best_value = value
            self.no_improve_cnt = 0
        else:
            self.no_improve_cnt += 1
        print(self.no_improve_cnt)
        return value

    def h(self, node):
        # all the cities left
        cities = []
        for i in range(len(self.cities)):
            if i in list(node.state):
                continue
            cities.append(i)
        cities.append(self.initial[0])

        #print(cities)
        # minimum spanning tree
        num = len(cities)
        dis_mat = np.zeros((len(self.cities), len(self.cities)))
        for i in range(num):
            for j in range(num):
                if i != j:
                    dis_mat[i, j] = np.sqrt((self.cities[cities[i]][0] - self.cities[cities[j]][0])**2 + (self.cities[cities[i]][1] - self.cities[cities[j]][1])**2)

        X = csr_matrix(dis_mat)
        Tcsr = minimum_spanning_tree(X)
        #print(Tcsr.toarray())

        return np.sum(np.sum(Tcsr))


def generate_random_cities(city_num=5, length=100):
    cities = []
    xs = []
    ys = []
    for i in range(city_num):
        x, y = random.uniform(0, length), random.uniform(0, length)
        cities.append((x, y))
        xs.append(x)
        ys.append(y)
    '''
    plt.plot(xs, ys, 'ro')
    plt.axis([-1, length+1, -1, length+1])
    plt.show()
    '''
    return cities


def generate_random_init(city_number=5):
    init= random.sample(range(city_number), city_num)
    return tuple(init)


def value_function(b, n, d):
    return math.pow(b, d + 1) - (n + 1) * b + n

city_num = 8
start = generate_random_init(city_num)
print(start)

tsp = TSPHillClimbingProblem(start, cities=generate_random_cities(city_num, 100))
print(tsp.actions(start))

# compare the execution time
start_time = time.time()
path = hill_climbing(tsp)
print("Execution time for A*: {}".format(time.time() - start_time))


xs = []
ys = []
for i in range(len(path)):
    xs.append(tsp.cities[path[i]][0])
    ys.append(tsp.cities[path[i]][1])
xs.append(tsp.cities[path[0]][0])
ys.append(tsp.cities[path[0]][1])

plt.plot(xs, ys)
plt.axis([-1, 100. + 1, -1, 100 + 1])
plt.show()
