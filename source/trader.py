import abc
from exchange import StockExchange
from qlearner import TabularQMatrix
from sarsa import TabularSarsaMatrix, RandomForestSarsaMatrix, GbmSarsaMatrix, SvrSarsaMatrix

class StockTrader(abc.ABC):
    ''' A stock trader that wraps a q-learning or sarsa learner inside
    Attributes:
        name (str): name of this trader
        utility (float): the utility constant
        exchange (StockExchange): reference an exchange to send orders to
        holding (int): the most recent number of shares in holding
        transaction_cost (float): cost of the latest trade it placed
        wealth (float): the most recent cumulative wealth
        step_count (int): the number of iterations it has completed
        reward (float): the most recent reward received
        state (tuple): the most recent state of the environment seen
        learner (Learner): the internal learner (q-learning or sarsa)
    '''

    def __init__(self, name: str, utility: float, exchange: StockExchange):
        self.name = name
        assert utility > 0
        self.utility = utility
        self.exchange = exchange
        self.exchange.register_trader(self)
        self.holding = 0
        self.transaction_cost = 0
        self.wealth = 0
        self.step_count = 0
        self.reward = None
        self.state = (self.exchange.curr_price, self.holding)
        self.learner = None
    
    def reset_episode(self):
        ''' Reset at the beginning of an episode
        '''
        self.holding = 0
        self.transaction_cost = 0
        self.wealth = 0
        self.step_count = 0
        self.reward = None
        self.state = (self.exchange.curr_price, self.holding)
        self.learner.reset_episode()

    def get_updated_price(self, old_price: float, new_price: float):
        ''' Observe that the stock price on the exchange has moved
        Args:
            old_price (float): the previous stock price before one-step simulation
            new_price (float): the new stock price after one-step simulation
        '''
        pnl = self.holding * (new_price - old_price)
        delta_wealth = pnl - self.transaction_cost
        self.wealth += delta_wealth
        self.step_count += 1
        self.reward = delta_wealth - 0.5 * self.utility * (delta_wealth - self.wealth / self.step_count)**2
        self.state = (new_price, self.holding)
    
    def place_order(self):
        ''' Find an order from internal learner. Send this to exchange to execute.
        '''
        order = self.learner.learn(self.reward, self.state)
        # adjust the order according to exchange.max_holding
        if self.holding + order > self.exchange.max_holding:
            order = self.exchange.max_holding - self.holding
        if self.holding + order < -self.exchange.max_holding:
            order = -self.exchange.max_holding - self.holding

        self.transaction_cost = self.exchange.execute(order)
        self.holding += order

class TabularQMatrixStockTrader(StockTrader):
    ''' A stock trader whose internal learner is tabular q-learning
    '''
    def __init__(self, name: str, utility: float, exchange: StockExchange,
    actions: tuple, epsilon: float, learning_rate: float, discount_factor: float):
        super().__init__(name, utility, exchange)
        self.learner = TabularQMatrix(actions, epsilon, learning_rate, discount_factor)

class TabularSarsaStockTrader(StockTrader):
    ''' A stock trader whose internal learner is tabular sarsa
    '''
    def __init__(self, name: str, utility: float, exchange: StockExchange,
    actions: tuple, epsilon: float, learning_rate: float, discount_factor: float):
        super().__init__(name, utility, exchange)
        self.learner = TabularSarsaMatrix(actions, epsilon, learning_rate, discount_factor)

class RFSarsaStockTrader(StockTrader):
    ''' A stock trader whose internal learner is random forest sarsa matrix
    '''
    def __init__(self, name: str, utility: float, exchange: StockExchange,
    actions: tuple, epsilon: float, learning_rate: float, discount_factor: float):
        super().__init__(name, utility, exchange)
        self.learner = RandomForestSarsaMatrix(actions, epsilon, learning_rate, discount_factor)

class GbmSarsaStockTrader(StockTrader):
    ''' A stock trader whose internal learner is sarsa matrix with gradient boosting
    '''
    def __init__(self, name: str, utility: float, exchange: StockExchange,
    actions: tuple, epsilon: float, learning_rate: float, discount_factor: float):
        super().__init__(name, utility, exchange)
        self.learner = GbmSarsaMatrix(actions, epsilon, learning_rate, discount_factor)

class SvrSarsaStockTrader(StockTrader):
    ''' A stock trader whose internal learner is sarsa matrix with SVR
    '''
    def __init__(self, name: str, utility: float, exchange: StockExchange,
    actions: tuple, epsilon: float, learning_rate: float, discount_factor: float):
        super().__init__(name, utility, exchange)
        self.learner = SvrSarsaMatrix(actions, epsilon, learning_rate, discount_factor)