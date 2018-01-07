from exchange import StockExchange

class StockTrader(object):
    ''' A stock trader that wraps a learner inside
    '''

    def __init__(self, exchange: StockExchange):
        self.exchange = exchange
        exchange.register_trader(self)