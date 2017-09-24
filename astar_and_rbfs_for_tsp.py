from search import Problem, astar_search, recursive_best_first_search
from utils import is_in
import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class TSPProblem(Problem):

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
        self.citiy_num = len(cities)
        Problem.__init__(self, initial, goal)

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        actions = []
        if 255 in list(state):
            none_pos = state.index(255)
            visited_cities = list(state)[0:none_pos+1]

            for i in range(len(self.cities)):
                if i in set(visited_cities):
                    continue
                actions.append(i)
        else:
            actions.append(0)
        return actions

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        next_state = list(state)
        none_pos = next_state.index(255)
        next_state[none_pos] = action

        return tuple(next_state)

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        return 255 not in list(state)

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        pos = list(state1).index(255)

        x = self.cities[state1[pos - 1]][0] - self.cities[action][0]
        y = self.cities[state1[pos - 1]][1] - self.cities[action][1]

        return c + math.sqrt(x*x + y*y)

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        pass

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
    init = [255] * city_number
    init[0] = random.choices(range(city_number))[0]
    return tuple(init)


def value_function(b, n, d):
    return math.pow(b, d + 1) - (n + 1) * b + n


def effective_branching_factor(nodes_num, depth):
    left = 1.05
    right = 4

    error_middle = 100
    while abs(error_middle) > 0.01:
        middle = (left + right) / 2
        error_middle = value_function(middle, nodes_num, depth)
        if error_middle > 0:
            right = middle
        else:
            left = middle

    return middle

#print(effective_branching_factor(52, 5))
city_num = 10
start = generate_random_init(city_num)
print(start)

tsp = TSPProblem(start, cities=generate_random_cities(city_num, 100))

# compare the execution time
start_time = time.time()
node_astar = astar_search(tsp)
depth = len(node_astar.path())
print("Execution time for A*: {}".format(time.time() - start_time))
print("Node number: {}".format(tsp.nodes_cnt))
print("Depth: {}".format(depth))
print("Effective branching factor: {}".format(effective_branching_factor(tsp.nodes_cnt, depth)))
tsp.nodes_cnt = 0
start_time = time.time()
node_rbfs = recursive_best_first_search(tsp)
depth = len(node_rbfs.path())
print("Execution time for RBFS: {}".format(time.time() - start_time))
print("Node number: {}".format(tsp.nodes_cnt))
print("Depth: {}".format(depth))
print("Effective branching factor: {}".format(effective_branching_factor(tsp.nodes_cnt, depth)))




print(node_astar.state)

xs = []
ys = []
for i in range(len(node_astar.state)):
    xs.append(tsp.cities[node_astar.state[i]][0])
    ys.append(tsp.cities[node_astar.state[i]][1])
xs.append(tsp.cities[node_astar.state[0]][0])
ys.append(tsp.cities[node_astar.state[0]][1])

plt.plot(xs, ys)
plt.axis([-1, 100. + 1, -1, 100 + 1])
plt.show()

'''

goal = (255, 1, 2, 3, 4, 5, 6, 7, 8)
eight_puzzle = EightPuzzleProblem(start, goal, random_h=False)

if eight_puzzle.reachable:
    # compare the execution time
    start_time = time.time()
    node_astar = astar_search(eight_puzzle)
    depth = len(node_astar.path())
    print("Execution time for A*: {}".format(time.time() - start_time))
    print("Node number: {}".format(eight_puzzle.nodes_cnt))
    print("Depth: {}".format(depth))
    print("Effective branching factor: {}".format(effective_branching_factor(eight_puzzle.nodes_cnt, depth)))

    eight_puzzle.nodes_cnt = 0
    start_time = time.time()
    node_rbfs = recursive_best_first_search(eight_puzzle)
    depth = len(node_rbfs.path())
    print("Execution time for RBFS: {}".format(time.time() - start_time))
    print("Node number: {}".format(eight_puzzle.nodes_cnt))
    print("Depth: {}".format(depth))
    print("Effective branching factor: {}".format(effective_branching_factor(eight_puzzle.nodes_cnt, depth)))

    eight_puzzle = EightPuzzleProblem(start, goal, random_h=True)
    eight_puzzle.nodes_cnt = 0
    start_time = time.time()
    node_rbfs = recursive_best_first_search(eight_puzzle)
    depth = len(node_rbfs.path())
    print("Execution time for RBFS with random number: {}".format(time.time() - start_time))
    print("Node number: {}".format(eight_puzzle.nodes_cnt))
    print("Depth: {}".format(depth))
    print("Effective branching factor: {}".format(effective_branching_factor(eight_puzzle.nodes_cnt, depth)))
'''

'''
    node_tmp = node_astar
    path = [x.state for x in node_tmp.path()]
    actions = [x.action for x in node_tmp.path()]
    for i in range(len(path)):
        print("Action: {}".format(actions[i]))
        print(path[i][0:3])
        print(path[i][3:6])
        print(path[i][6:])
        print("-----------------------")
    print("Totally {} steps.".format(len(path)))
 '''

