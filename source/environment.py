import abc
from exchange import StockExchange

class Environment(abc.ABC):
    ''' Manage the interactions between a list of learners and exchanges
    Attributes:
        exchange (StockExchange): the exchange that references traders
    '''    

    def __init__(self, exchange: StockExchange):        
        self.exchange = exchange
    
    @abc.abstractmethod
    def run(self, nrun: int, report=False):
        ''' Run the learners for nrun iterations
        Args:
            nrun (int): number of iterations to run
            report (boolean): True to return a performance over time of each trader
        Returns:
            dict: map each trader to its performance over nrun steps
        '''
        raise NotImplementedError

class StockTradingEnvironment(Environment):
    # Override
    def run(self, nrun, report=False):
        self.exchange.reset_episode()
        if report is True:
            result = {trader.name: [trader.wealth] for trader in self.exchange.traders}
        for step_count in range(1,nrun+1):
            for trader in self.exchange.traders:
                trader.place_order()
            self.exchange.simulate_stock_price()

            if report is True:
                for trader in self.exchange.traders:
                    result[trader.name].append(trader.wealth)
            if step_count % 1000 == 0:
                print('finished {:,} runs'.format(step_count))
        if report is True:
            return result
        return None