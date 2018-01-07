from exchange import StockExchange

class StockTrader(object):
    ''' A stock trader that wraps a Q or Sarsa learner inside
    Attributes:
        name (str): name of this trader
        exchange (StockExchange): reference an exchange to send orders to
        price_observed (float): the latest stock price seen
        holding (int): current number of shares in holding
        last_cost (float): transaction cost of the latest trade it placed
        utility (float): the utility constant
        wealth (float): the current cumulative wealth
    '''

    def __init__(self, exchange: StockExchange):
        self.exchange = exchange
        exchange.register_trader(self)