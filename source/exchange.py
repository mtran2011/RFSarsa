from trader import StockTrader

class StockExchange(object):
    ''' An exchange referencing one single stock and a set of stock traders
    Stock price can only move in multiple of ticks
    Spread and impact costs are based on lot size and tick size
    Attributes:
        stock (Stock): the only stock on this exchange       
        lot (int): the lot size
        tick (int): the tick size for this single stock
        max_holding (int): the max number of shares a trader can long or short in cumulative
        traders (set): a set of instances of StockTrader
        prev_price (float): the one-step previous stock price rounded to tick
        curr_price (float): the current stock price rounded to tick
        roundings (dict): for convenience in rounding price to tick
    '''
    def __init__(self, stock: Stock, lot: int, tick: int, max_holding: int):
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
    
    def register_trader(self, trader: StockTrader):
        self.traders.add(trader)

    def reset_episode(self):
        # reset holding of each trader to 0
        # reset learner's last action and last state
        pass

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
    
    def notify_traders(self, price: float):
        # notify observers of new simulated stock price
        for trader in self.traders:
            trader.get_notified(price)