import abc
import random
import operator

class Learner(abc.ABC):
    ''' Abstract base class for a learning agent, either Q-learning or Sarsa
    Attributes:
        _actions (tuple): the list of all possible actions it can take
        _epsilon (float): constant used in epsilon greedy
        _count (int): number of learning steps it has done
        _last_action (object): the immediate previous action it took
        _last_state (tuple): to memorize the immediate previous state, for which it took _last_action
    '''
    def __init__(self, actions: tuple, epsilon: float):
        assert isinstance(actions, tuple) and (len(actions) > 0)
        self._actions = actions
        self._epsilon = epsilon
        self._count = 2 # to avoid divide by zero in log2(count)
        self._last_action = None
        self._last_state = None
    
    @abc.abstractmethod
    def _find_action_greedily(self, state: tuple, use_epsilon=True, return_q=False):
        ''' Given the state, find the best action using epsilon-greedy
        With probability of epsilon, pick a random action. Else pick a greedy action.
        Args:
            state (tuple): the given state which is a tuple of state attributes
            use_epsilon (bool): True if the randomization using epsilon is to be used
            return_q (bool): True if user wants to return the found value of Q(state, a)
        Returns: 
            object: the action found by epsilon-greedy
            float: the value Q(s,a) for state s found by epsilon-greedy
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, reward: float, new_state: tuple):
        ''' Get a reward and see a new_state. Use these to do some internal training if self._last_action is not None. 
        Then find and return a new action.        
        Update _last_state <- new_state
        Update _last_action <- best_action        
        Args:
            reward (float): the reward seen after the previous action
            new_state (tuple): the new_state seen after the previous action
        Returns:
            best_action (object): take a new action based on new_state
        '''
        raise NotImplementedError

    def reset_episode(self):
        ''' Reset at the beginning of an episode
        '''
        self._last_action = None
        self._last_state = None

class MatrixLearner(Learner):
    ''' Abstract base class for a Q-matrix or Sarsa-matrix
    Attributes:
        _Q (dict): dict of key tuple (s,a) to the float value Q(s,a)
        _learning_rate (float): the constant learning_rate
        _discount_factor (float): the constant discount_factor of future rewards
    '''
    def __init__(self, actions: tuple, epsilon: float, learning_rate: float, discount_factor: float):
        super().__init__(actions, epsilon)
        self._Q = dict()        
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
    
    @abc.abstractmethod
    def _get_q(self, state, action):
        ''' Find, or estimate, Q(s,a) for a given state s and action a
        Args:
            state (tuple): state s, a tuple of state attributes
            action (object): action a
        Returns:
            float: the value of Q(s,a). 
        '''
        raise NotImplementedError

    # Override
    def _find_action_greedily(self, state, use_epsilon=True, return_q=False):
        if use_epsilon and random.random() < self._epsilon:
            # with probability epsilon, choose from all actions with equal chance
            best_action = random.choice(self._actions)
            max_q = None
            if return_q:
                max_q = self._get_q(state, best_action)
        else:
            # choose action = arg max {action} of Q(state, action)
            q_values = [(self._get_q(state, action), action) for action in self._actions]
            max_q, best_action = max(q_values, key=operator.itemgetter(0))
            # if max_q is 0, then either this state has never been visited
            # or the state has been visited but previous action results in negative reward
            if max_q == 0:
                best_action = random.choice([pair[1] for pair in q_values if pair[0]==0])
        
        if return_q:
            return best_action, max_q
        return best_action