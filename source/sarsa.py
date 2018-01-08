import abc
from math import log2
import numpy as np
from learner import Learner, MatrixLearner
from sklearn.ensemble import RandomForestRegressor

class SarsaLearner(Learner):
    ''' Abstract base class for a Sarsa learner (on policy)
    '''

    @abc.abstractmethod
    def _train_internally(self, reward, next_q):
        ''' Train internal model using reward and next_q:=Q(s',a')
        The internal training can be:
            for matrix: update Q(_last_state, _last_action) += learning_rate * (reward + gamma * next_q - old_q)
            for SemiGrad: via grad descent, update the parameters in the function estimator
        Args:
            reward (float): the reward seen after taking self._last_action
            next_q (float): value of Q(s',a') where s' is the new_state, a' is new action based on epsilon-greedy (on policy)
        '''
        raise NotImplementedError

    # Override
    def learn(self, reward, new_state):
        action, next_q = self._find_action_greedily(new_state, use_epsilon=True, return_q=True)
        if self._last_action is not None and self._last_state is not None:
            self._train_internally(reward, next_q)
        self._last_action = action
        self._last_state = new_state
        self._count += 1
        self._epsilon = min(self._epsilon, 1 / log2(self._count))
        return action

class SarsaMatrix(MatrixLearner, SarsaLearner):
    ''' Abstract class for a Sarsa-learner matrix that holds the values of Q(s,a)
    '''
    # Override
    def _train_internally(self, reward, next_q):
        if self._last_action is None or self._last_state is None:
            return None
        old_q = self._get_q(self._last_state, self._last_action)
        new_q = old_q + self._learning_rate * (reward + self._discount_factor * next_q - old_q)
        self._Q[(self._last_state, self._last_action)] = new_q

class TabularSarsaMatrix(SarsaMatrix):
    ''' The discrete, tabular Sarsa-matrix learner
    '''
    # Override
    def _get_q(self, state, action):
        return self._Q.get((state, action), 0)

class RandomForestSarsaMatrix(SarsaMatrix):
    ''' Use random forest on all existing values of Q(s,a) to estimate new Q(s,a)
    The forest is refitted once every 500 steps to all of existing Q
    '''
    def __init__(self, actions, epsilon, learning_rate, discount_factor, max_nfeatures=2):
        super().__init__(actions, epsilon, learning_rate, discount_factor)
        self.randomforest = RandomForestRegressor(
            n_estimators=30, max_features=max_nfeatures,
            min_samples_leaf=5, n_jobs=2)
    
    # Override
    def _get_q(self, state, action):
        # in the first 500 training steps, do not estimate, just default to 0
        if not self._Q:
            return 0
        if (state, action) in self._Q:
            return self._Q[(state, action)]
        if self._count - 2 < 500:
            return 0
        # refit the random forest once every 500 steps
        if (self._count - 2) % 500 == 0:
            # prepare training data
            X, Y = [], []
            for key, value in self._Q.items():
                # key is a tuple of (s,a)
                X.append([*key[0], key[1]])
                Y.append(value)
            X = np.array(X)
            Y = np.array(Y)
            self.randomforest.fit(X, Y)
        # use the random forest to predict Q for this (state, action)
        x = np.array([*state, action]).reshape(1, len(state)+1)
        return np.asscalar(self.randomforest.predict(x))
