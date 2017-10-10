# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import operator
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        oldfood = currentGameState.getFood().asList()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        ghostposition = [ghost.getPosition() for ghost in newGhostStates]

        if newPos in ghostposition or action == 'Stop':
            return -float("inf")
        elif newPos in list(set(oldfood)-set(newFood)):
            return float("inf")
        elif newPos not in newFood:
            md_food_pos = [manhattanDistance(newPos, current_food) for current_food in newFood]
            return -min(md_food_pos)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        pacman_score, pacman_move = self.minimax(gameState, self.depth, True, self.index)
        return pacman_move

    def minimax(self, gameState, depth, player, agent):
        pacman_player = True
        ghost_player = False
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        if player == pacman_player:
            all_moves = gameState.getLegalActions()
            best_score_index, best_score = max(enumerate(
                [self.minimax(gameState.generateSuccessor(self.index,move), depth, False, 1) for move in all_moves]),
                key=operator.itemgetter(1))
            return best_score, all_moves[best_score_index]

        elif player == ghost_player:
            all_moves = gameState.getLegalActions(agent)
            if agent != gameState.getNumAgents()-1:
                worst_score_index, worst_score = min(enumerate(
                    [self.minimax(gameState.generateSuccessor(agent, move), depth, False, agent+1) for move in all_moves]),
                    key=operator.itemgetter(1))
            else:
                worst_score_index, worst_score = min(enumerate(
                    [self.minimax(gameState.generateSuccessor(agent, move), depth - 1, True, 0)[0] for move in all_moves]),
                    key=operator.itemgetter(1))
            return worst_score,all_moves[worst_score_index]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha, beta = -float("inf"), float("inf")
        pacman_move = self.alphabetaprune(gameState, 0, True, self.index, alpha, beta)
        return pacman_move

    def alphabetaprune(self, gameState, depth, player, agent, alpha, beta):
        pacman_player = True
        ghost_player = False
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        if player == pacman_player:
            best_score = -float("inf")
            score = best_score
            pacman_action = Directions.STOP
            all_moves = gameState.getLegalActions()
            for move in all_moves:
                score = self.alphabetaprune(gameState.generateSuccessor(self.index, move), depth, False, 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    pacman_action = move
                alpha = max(best_score, alpha)
                if best_score > beta:
                    return best_score
            if depth == 0:
                return pacman_action
            else:
                return best_score

        elif player == ghost_player:
            all_moves = gameState.getLegalActions(agent)
            best_score = float("inf")
            score = best_score
            for move in all_moves:
                if agent != gameState.getNumAgents() - 1:
                    score = self.alphabetaprune(gameState.generateSuccessor(agent, move), depth, False, agent + 1, alpha, beta)
                else:
                    if depth == self.depth -1:
                        score = self.evaluationFunction(gameState.generateSuccessor(agent, move))
                    else:
                        score = self.alphabetaprune(gameState.generateSuccessor(agent, move), depth + 1, True, 0, alpha, beta)
                if score < best_score:
                    best_score = score
                beta = min(best_score, beta)
                if best_score < alpha:
                    return best_score
            return best_score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        pacman_move, pacman_score = self.expectimax(gameState, 0, True, self.index)
        return pacman_move

    def expectimax(self, gameState, depth, player, agent):
        pacman_player = True
        ghost_player = False
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if player == pacman_player:
            best_score, best_move = -float("inf"), ""
            all_moves = gameState.getLegalActions()
            for move in all_moves:
                if move == 'Stop':
                    continue
                maxscore = self.expectimax(gameState.generateSuccessor(self.index, move), depth, False, 1)
                if type(maxscore) is tuple:
                    maxscore = [x for x in maxscore if type(x) is float][0]
                if maxscore > best_score:
                    best_score = maxscore
                    best_move = move
            return best_move, best_score

        elif player == ghost_player:
            best_move = ""
            all_moves = gameState.getLegalActions(agent)
            prob = 1.0 / len(all_moves)
            minscores = 0.0
            for move in all_moves:
                if move == 'Stop':
                    continue
                if agent != gameState.getNumAgents() - 1:
                    minscores = self.expectimax(gameState.generateSuccessor(agent, move), depth, False, agent + 1)
                    if type(minscores) is tuple:
                        minscores = [x for x in minscores if type(x) is float][0]
                else:
                    minscores = self.expectimax(gameState.generateSuccessor(agent, move), depth + 1, True, 0)
                if type(minscores) is tuple:
                    minscores = [x for x in minscores if type(x) is float][0]

                minscores += minscores*prob
                best_move = move
            return best_move, minscores


        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return -float("inf")

    pacman_pos = currentGameState.getPacmanPosition()
    capsules_left = len(currentGameState.getCapsules())
    all_food = currentGameState.getFood().asList()
    food_left = len(all_food)
    md_closest_food = min([util.manhattanDistance(pacman_pos, food) for food in all_food])

    scared_ghost, active_ghost =[],[]
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer:
            scared_ghost.append(ghost)
        else:
            active_ghost.append(ghost)

    dist_nearest_scaredghost = dist_nearest_activeghost = 0

    if not len(scared_ghost):
        dist_nearest_scaredghost = 0

    if not len(active_ghost):
        dist_nearest_activeghost = float("inf")

    if active_ghost:
        dist_nearest_activeghost = min([util.manhattanDistance(pacman_pos, ghost.getPosition()) for ghost in active_ghost])
        if dist_nearest_activeghost > 10:
            dist_nearest_activeghost = 10
    if scared_ghost:
        dist_nearest_scaredghost = min([util.manhattanDistance(pacman_pos, ghost.getPosition()) for ghost in scared_ghost])

    return currentGameState.getScore() + -1*md_closest_food + 2*(1.0/dist_nearest_activeghost)+ 3*dist_nearest_scaredghost+ -20*capsules_left + -4*food_left

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

