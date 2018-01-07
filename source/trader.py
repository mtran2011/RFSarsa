from exchange import StockExchange

class StockTrader(object):
    ''' A stock trader that wraps a learner inside
    Attributes:
        exchange (StockExchange): reference an exchange to send orders to
        last_price (float): the latest stock price seen
        holding (int): current number of shares in holding
        last_cost (float): transaction cost of the latest trade it placed
        utility (float): the utility constant
        wealth (float): the current cumulative wealth
        

    '''

    def __init__(self, exchange: StockExchange):
        self.exchange = exchange
        exchange.register_trader(self)