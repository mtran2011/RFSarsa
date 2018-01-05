import abc

class Environment(abc.ABC):
    ''' Manage the interactions between a list of learners and exchanges
    Attributes:        
        learners (list): list of Learners
        exchange (StockExchange): the exchange
    '''    

    def __init__(self, learners: list, exchange: StockExchange):
        self.learners = learners
        self.exchange = exchange
    
    @abc.abstractmethod
    def run(self, util, nrun, report=False):
        ''' Run the learners for nrun iterations
        Args:
            util (float): the constant in the utility function
            nrun (int): number of iterations to run
            report (boolean): True to return a list of performance over time
        Returns:
            list: list of performance, e.g., cumulative wealth
        '''
        pass

class StockTradingEnvironment(Environment):
    # Override
    def run(self, util, nrun, report=False):
        # reset last_action and last_state of learner to None
        # so that it doesn't use the initial reward=0 to learn internally
        # reset shareholdings in exchange to 0
        for learner in self.learners:
            learner.reset_last_action()
        self.exchange.reset_episode()

        reward = 0
        state = (self.exchange.report_stock_price(), self.exchange.num_shares_owned)
        wealth, wealths = 0, []

        for iter_ct in range(1,nrun+1):
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