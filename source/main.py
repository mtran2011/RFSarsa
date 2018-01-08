import itertools
from math import log
import pandas as pd
import matplotlib.pyplot as plt
from stock import OULogStock
from exchange import StockExchange
from trader import TabularQMatrixStockTrader, TabularSarsaStockTrader, RFSarsaStockTrader
from environment import StockTradingEnvironment

def graph_performance(df: pd.DataFrame, ntrain: int, version: int):
    linestyles = itertools.cycle(['-', ':', '--', '-.'])
    colors = itertools.cycle(['b', 'g', 'r', 'y', 'k', 'm', 'c'])
    plt.figure()
    for name in df.columns:
        plt.plot(df.index, df[name], label=name, linestyle=next(linestyles), color=next(colors))
    ntest = len(df.index)-1
    plt.title('Performance with ntrain = {0:,} and ntest = {1:,}'.format(ntrain, ntest))
    plt.legend(loc='best')
    plt.xlabel('iterations in the test run')
    plt.ylabel('cumulative wealth')
    plt.tight_layout()
    plt.savefig('../figs/newfig{}.png'.format(version))

def run_stock_trading(version: int):
    oustock = OULogStock(price=50, maxp=1050, minp=0, kappa=0.1, mu=log(75), sigma=0.1)
    lot = 10
    actions = tuple(range(-5*lot, 6*lot, lot))
    stock_exchange = StockExchange(oustock, lot, tick=1, max_holding=100*lot)
    utility, ntrain, ntest = 1e-3, 5000, 1000
    epsilon, learning_rate, discount_factor = 0.1, 0.5, 0.999
    tabular_qmatrix = TabularQMatrixStockTrader('tabular q-learning', utility, stock_exchange, actions, epsilon, learning_rate, discount_factor)
    tabular_sarsa = TabularSarsaStockTrader('tabular sarsa', utility, stock_exchange, actions, epsilon, learning_rate, discount_factor)
    rf_sarsa = RFSarsaStockTrader('random forest sarsa', utility, stock_exchange, actions, epsilon, learning_rate, discount_factor)
    trading_environment = StockTradingEnvironment(stock_exchange)
    trading_environment.run(ntrain)
    result = trading_environment.run(ntest, report=True)
    result = pd.DataFrame(result)
    result.sort_index(axis=1, inplace=True)
    graph_performance(result, ntrain, version)

def main():
    for version in range(12):
        run_stock_trading(version)

if __name__ == '__main__':
    main()
    