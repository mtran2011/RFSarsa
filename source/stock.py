import abc
import random
from math import exp, log

class Stock(abc.ABC):
    ''' Abstract base class for a stock object
    Attributes:
        price (float): the current spot price in USD
        maxp (float): the max price this stock can reach
        minp (float): the lowest price this stock can reach, subject to 0
    '''
    
    def __init__(self, price: float, maxp: float, minp: float):
        assert minp < price < maxp
        assert minp >= 0
        self.price = price
        self.maxp = maxp
        self.minp = minp

    @abc.abstractmethod
    def simulate_price(self, dt=1.0):
        ''' Simulate self.price over time step dt based on some internal models
        Args:
            dt (float): length of time step
        Returns:
            float: the new updated price
        '''
        raise NotImplementedError

class OULogStock(Stock):
    ''' Stock with dlogS following an OU process
    dlogS = kappa * (mu - logS) * dt + sigma * dW where var(dW) = dt    
    '''
    def __init__(self, price: float, maxp: float, minp: float,
    kappa: float, mu: float, sigma: float):
        assert kappa >= 0 and sigma >= 0
        super().__init__(price, maxp, minp)
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma
    
    # Override
    def simulate_price(self, dt=1.0):
        if self.price == 0:
            return 0
        old_log = log(self.price)
        dW = dt**0.5 * random.gauss(0,1)
        dlogS = self.kappa * (self.mu - old_log) * dt + self.sigma * dW
        self.price = self.price * exp(dlogS)
        self.price = max(self.price, self.minp)
        self.price = min(self.price, self.maxp)
        return self.price