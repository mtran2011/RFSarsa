from stock import Stock

class StockExchange(object):
    ''' An exchange referencing one single stock and a set of stock traders
    Stock price can only move in multiple of ticks
    Spread and impact costs are based on lot size and tick size
    Attributes:
        stock (Stock): the only stock on this exchange       
        lot (int): the lot size
        tick (float): the tick size for this single stock
        max_holding (int): the max number of shares a trader can long or short in cumulative
        traders (set): a set of instances of StockTrader
        prev_price (float): the one-step previous stock price rounded to tick
        curr_price (float): the current stock price rounded to tick
        roundings (dict): for convenience in rounding price to tick
    '''
    def __init__(self, stock: Stock, lot: int, tick: float, max_holding: int):
        self.roundings = {1: 0, 0.1: 1, 0.01: 2}        
        assert lot > 0 and max_holding > 0
        assert tick in self.roundings
        self.stock = stock
        self.lot = lot
        self.tick = tick
        self.max_holding = max_holding
        self.traders = set()
        self.prev_price = None
        self.curr_price = round(stock.price, self.roundings[tick])
    
    def register_trader(self, trader):
        ''' Register a trader
        '''
        self.traders.add(trader)

    def reset_episode(self):
        ''' Reset at the beginning of an episode
        '''
        self.prev_price = None
        for trader in self.traders:
            trader.reset_episode()

    def execute(self, order: int):
        ''' Execute an order from a particular trader
        Args:
            order (int): how many shares to buy (positive) or sell (negative)
        Returns:
            float: transaction cost
        '''
        num_lots = abs(order) / self.lot
        spread_cost = num_lots * self.tick
        impact_cost = num_lots**2 * self.tick
        return spread_cost + impact_cost
    
    def notify_traders(self, old_price: float, new_price: float):
        ''' Notify observers of new simulated stock price
        '''
        for trader in self.traders:
            trader.get_updated_price(old_price, new_price)
    
    def simulate_stock_price(self):
        ''' Simulate the internal stock for one time step
        '''
        self.prev_price = self.curr_price
        self.curr_price = round(self.stock.simulate_price(), self.roundings[self.tick])
        self.notify_traders(self.prev_price, self.curr_price)