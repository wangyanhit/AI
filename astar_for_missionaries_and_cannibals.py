from search import Problem, astar_search
from utils import is_in


class MCProblem(Problem):

    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.states = [(3, 3, 0), (3, 0, 0), (2, 3, 0), (2, 2, 0), (2, 0, 0), (1, 3, 0), (1, 1, 0), (1, 0, 0), (0, 3, 0), (0, 0, 0),
                       (3, 3, 1), (3, 0, 1), (2, 3, 1), (2, 2, 1), (2, 0, 1), (1, 3, 1), (1, 1, 1), (1, 0, 1), (0, 3, 1), (0, 0, 1)]
        Problem.__init__(self, initial, goal)

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        next_states = []
        state_c, state_m, state_b = state
        for next_state in self.states:
            next_state_c, next_state_m, next_state_b = next_state
            if state_b == 0:
                if state_b == next_state_b:
                    continue
                if next_state_c > state_c or next_state_m > state_m:
                    continue
                error = -next_state_c - next_state_m + state_c + state_m
                if error > 2 or error < 1:
                    continue
            else:
                if state_b == next_state_b:
                    continue
                if next_state_c < state_c or next_state_m < state_m:
                    continue
                error = next_state_c + next_state_m - state_c - state_m
                if error > 2 or error < 1:
                    continue
            next_states.append(next_state)
        return next_states

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
        goal_state_c, goal_state_m, _ = self.goal
        state_c, state_m, _ = node.state
        return 0.5 * (goal_state_c + goal_state_m - state_c - state_m)

mc = MCProblem((3, 3, 0), (0, 0, 1))
node = astar_search(mc)
node_tmp = node
path = [x.state for x in node.path()]
for i in range(len(path)):
    print(path[i])
