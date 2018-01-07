import abc
from math import log2
from learner import Learner, MatrixLearner

# Adapted from:
# github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial1/qlearn.py

class QLearner(Learner):
    ''' Abstract base class for a Q-learning agent (off policy)
    '''
    
    @abc.abstractmethod
    def _train_internally(self, reward, new_state):
        ''' Use reward and new_state to train the internal model.
        The internal training can be:
            for Q matrix: update Q(_last_state, _last_action) += (reward + gamma * maxQ(new_state,action) - old_q) * learning_rate
            for DQN: add the experience (s,a,r,s') to memory and train the internal neural network
            for SemiGradQLearner: via grad descent, update the parameters in the function estimator
        Args:
            reward (float): the reward seen after taking self._last_action
            new_state (tuple): the new_state seen after taking self._last_action
        '''
        raise NotImplementedError

    # Override
    def learn(self, reward, new_state):
        # if this agent has taken at least one any action before
        if self._last_action is not None and self._last_state is not None:
            self._train_internally(reward, new_state)
        action = self._find_action_greedily(new_state, use_epsilon=True, return_q=False)
        self._last_action = action
        self._last_state = new_state
        self._count += 1
        self._epsilon = min(self._epsilon, 1 / log2(self._count))
        return action

class QMatrix(MatrixLearner, QLearner):
    ''' Abstract class for a Q-learner matrix that holds the values of Q(s,a)    
    '''
    # Override
    def _train_internally(self, reward, new_state):
        if self._last_action is None or self._last_state is None:
            return None
        old_q = self._get_q(self._last_state, self._last_action)
        _, max_q = self._find_action_greedily(new_state, use_epsilon=False, return_q=True)
        new_q = old_q + self._learning_rate * (reward + self._discount_factor * max_q - old_q)
        self._Q[(self._last_state, self._last_action)] = new_q

class TabularQMatrix(QMatrix):
    ''' The discrete, tabular Q-matrix learner
    '''
    # Override
    def _get_q(self, state, action):
        return self._Q.get((state, action), 0)
    