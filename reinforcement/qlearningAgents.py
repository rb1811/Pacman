# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.Qval = util.Counter()
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if (state, action) not in self.Qval: #If a new state return 0.0
            self.Qval[(state, action)] = 0.0
        return self.Qval[(state, action)] #Else return the Q value.
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        if not len(self.getLegalActions(state)):#If no legal actions return 0.0
            return 0.0
        else:
            all_actions = util.Counter()
            for action in self.getLegalActions(state): #For all actions possible for that state return the max of all actions
                all_actions[action] = self.getQValue(state, action)
            return all_actions[all_actions.argMax()]
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        if not len(self.getLegalActions(state)): #If its a terminal state return None
            return None
        else:#Else for all actions possible for that state return action that gives you the maximum qvalue
            all_actions = util.Counter()
            for action in self.getLegalActions(state):
                all_actions[action] = self.getQValue(state, action)
            return all_actions.argMax()
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None #Default action is none and is returned if no legal actions are possible for the state.
        if len(legalActions):
            if util.flipCoin(self.epsilon): # With an epsilon probability explore
                return random.choice(legalActions)
            else: #with 1-epsilon probability exploit
                return self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          N
          OTE: You should never call this function,
          it will be called on your behalf
        """
        #Use the formula to compute the Q-values
        self.Qval[(state, action)] = (1-self.alpha)*self.getQValue(state, action) + self.alpha*(reward + self.discount * self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights #return the weight

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        Qvalue = 0
        feature_list = self.featExtractor.getFeatures(state, action)
        for key in feature_list.keys(): #Compute the q-values of the state based on the weights and features for the state
            Qvalue += self.weights[key]*feature_list[key]
        return Qvalue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features_list =  self.featExtractor.getFeatures(state, action) #Get all the features possible for the state.
        if not len(self.getLegalActions(nextState)): #Update the weights based on the value
            diff = reward - self.getQValue(state, action)
        else:
            diff = (reward + self.discount * max([self.getQValue(nextState, nextaction) for nextaction in
                      self.getLegalActions(nextState)])) - self.getQValue(state, action) #Based on the formula compute the differences.
        for key in features_list.keys():
            self.weights[key] +=  self.alpha * diff * features_list[key] #update the weights based on the differences.
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
