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

        # if report, tell all traders to memorize performance

        for iter_ct in range(1,nrun+1):
            for trader in self.exchange.traders:
                trader.place_order()
            self.exchange.simulate_stock_price()

            
            
            order = self.learner.learn(reward, state)
            transaction_cost = self.exchange.execute(order)
            pnl = self.exchange.simulate_stock_price()

            delta_wealth = pnl - transaction_cost
            wealth += delta_wealth

            reward = delta_wealth - 0.5 * util * (delta_wealth - wealth / iter_ct)**2
            state = (self.exchange.report_stock_price(), self.exchange.num_shares_owned)

            if report:
                wealths.append(wealth)
            if iter_ct % 1000 == 0:
                print('finished {:,} runs'.format(iter_ct))
        
        if report:
            return wealths
        else:
            return None