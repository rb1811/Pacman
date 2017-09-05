# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import time
from collections import OrderedDict


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    time.sleep(10)
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print  "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    fringe = util.Stack()
    fringe.push(problem.getStartState())
    closed_list, child= [], {True:problem.getStartState()}
    while True:
        if fringe.isEmpty():
            return False
        parent = fringe.pop()
        closed_list.append(parent)
        if problem.isGoalState(parent):
            return generate_path(parent, problem, child)
        for successor in problem.getSuccessors(parent):
            if successor[0] not in closed_list:
                fringe.push(successor[0])
                child[successor[0]] = [parent, successor[1]]
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    fringe = util.Queue()
    fringe.push(problem.getStartState())
    closed_list, child = [problem.getStartState()],  {True: problem.getStartState()}
    while True:
        if fringe.isEmpty():
            return False
        parent = fringe.pop()
        if problem.isGoalState(parent):
            return generate_path(parent, problem, child)
        # closed_list.append(parent)
        for successor in problem.getSuccessors(parent):
            if successor[0] not in closed_list and successor[0] not in fringe.list:
                closed_list.append(successor[0])
                fringe.push(successor[0])
                child[successor[0]] = [parent, successor[1]]
    util.raiseNotDefined()


def uniformCostSearch(problem):
    # """Search the node of least total cost first."""
    fringe = util.PriorityQueue()
    cumulative_cost = 0
    fringe.update(problem.getStartState(), cumulative_cost)
    closed_list, child, cost = [], {True: problem.getStartState()}, {problem.getStartState(): 0}
    while True:
        if fringe.isEmpty():
            return False
        parent = fringe.pop()
        if parent in closed_list:
            continue
        cumulative_cost = cost[parent]
        if problem.isGoalState(parent):
            return generate_path(parent, problem, child)
        closed_list.append(parent)
        for successor in problem.getSuccessors(parent):
            if cost.has_key(successor[0]):
                if cost[successor[0]] > cumulative_cost+successor[2]:
                    cost[successor[0]] = cumulative_cost+successor[2]
                    child[successor[0]] = [parent, successor[1]]
            else:
                cost[successor[0]] =  cumulative_cost+successor[2]
                child[successor[0]] =  [parent, successor[1]]
            fringe.update(successor[0], cumulative_cost+successor[2])

    util.raiseNotDefined()


def generate_path(parent, problem, child):
    path = []
    key = parent
    while key is not problem.getStartState():
        path.append(child[key][1])
        key = child[key][0]
    path = path[::-1]
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue()
    cumulative_cost = 0
    fringe.push(problem.getStartState(), cumulative_cost)
    closed_list, child, cost = [], {True: problem.getStartState()}, {problem.getStartState(): 0+heuristic(problem.getStartState(), problem)}
    while True:
        if fringe.isEmpty():
            return False
        parent = fringe.pop()
        if parent in closed_list:
            continue
        cumulative_cost = cost[parent]
        if problem.isGoalState(parent):
            return generate_path(parent, problem, child)
        closed_list.append(parent)
        for successor in problem.getSuccessors(parent):
            if cost.has_key(successor[0]):
                if cost[successor[0]] > cumulative_cost+successor[2]:
                    cost[successor[0]] = cumulative_cost+successor[2]
                    child[successor[0]] = [parent, successor[1]]
            else:
                cost[successor[0]] =  cumulative_cost+successor[2]
                child[successor[0]] =  [parent, successor[1]]
            fringe.update(successor[0], cumulative_cost+successor[2]+heuristic(successor[0],problem))
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
