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
    '''
    def __init__(self, stock: Stock, lot: int, tick: int, max_holding: int):
        if lot <= 0 or max_holding <= 0:
            raise ValueError('lot and max_holding must be positive')
        if tick not in (1, 0.1, 0.01):
            raise ValueError('tick must be 1, 0.1, or 0.01')
        self.stock = stock
        self.lot = lot
        self.tick = tick
        self.max_holding = max_holding
        self.traders = set()
    
    def register_trader(self, trader: StockTrader):
        self.traders.add(trader)

    def reset_traders_holdings(self):
        ''' To reset before a new game episode
        '''
        for trader in self.traders:
            trader.reset_holding()

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
        # notify observers of new stock price