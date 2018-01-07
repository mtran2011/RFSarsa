from exchange import StockExchange

class StockTrader(object):
    ''' A stock trader that wraps a Q or Sarsa learner inside
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
    
    def reset_episode(self):
        ''' Reset at the beginning of an episode
        '''
        pass

    def get_updated_price(self, old_price: float, new_price: float):
        ''' Observe that the stock price on the exchange has moved
        '''
        pass
    