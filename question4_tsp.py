from search import Problem, astar_search, recursive_best_first_search
from utils import is_in
import random
import time
import math


class TSPProblem(Problem):

    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""
    def __init__(self, initial, goal=None, random_h=False):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.nodes_cnt = 0
        Problem.__init__(self, initial, goal)

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        actions = []
        none_pos = state.index(255)
        none_row = int(none_pos / 3)
        none_col = int(none_pos % 3)
        if none_row != 0:
            actions.append('U')
        if none_row != 2:
            actions.append('D')
        if none_col != 0:
            actions.append('L')
        if none_col != 2:
            actions.append('R')
        return actions

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        next_state = list(state)
        none_pos = next_state.index(255)
        if action == 'U':
            tmp = next_state[none_pos]
            next_state[none_pos] = next_state[none_pos - 3]
            next_state[none_pos - 3] = tmp
        elif action == 'D':
            tmp = next_state[none_pos]
            next_state[none_pos] = next_state[none_pos + 3]
            next_state[none_pos + 3] = tmp
        elif action == 'L':
            tmp = next_state[none_pos]
            next_state[none_pos] = next_state[none_pos - 1]
            next_state[none_pos - 1] = tmp
        elif action == 'R':
            tmp = next_state[none_pos]
            next_state[none_pos] = next_state[none_pos + 1]
            next_state[none_pos + 1] = tmp

        return tuple(next_state)

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

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
        pass

    def h(self, node):
        sum_d = 0
        for index, item in enumerate(node.state):
            if item != 255:
                goal_pos = self.goal.index(item)
                goal_pos_row = int(goal_pos / 3)
                goal_pos_col = int(goal_pos % 3)
                cur_pos_row = int(index / 3)
                cur_pos_col = int(index % 3)
                sum_d += abs(goal_pos_col - cur_pos_col) + abs(goal_pos_row - cur_pos_row)
        if self.random_h:
            sum_d += random.choices([0, 1, 2, 3])[0]
        return sum_d


def generate_random_init():
    num = list([1, 2, 3, 4, 5, 6, 7, 8, 255])
    random.shuffle(num)
    return tuple(num)


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

start = generate_random_init()
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