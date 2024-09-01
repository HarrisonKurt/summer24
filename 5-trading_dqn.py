from helpers.data import Data
from helpers.trading_dqn import TradingDQN

symbol = "VTI"
training_filename = f"./data/{symbol}.csv"
sample_filename = f"./data/{symbol}-latest.csv"

Xtrain, Xval = Data.split_data(Data.read_data(training_filename))
Xsample = Data.read_data(sample_filename)

iterations = 5000
sell_all = True
dqn = TradingDQN(10, 0.10, 0.05, 0.05)
dqn.train(symbol, Xtrain, Xval, iterations, sell_all)
profit, data = dqn.run(symbol, iterations, Xsample, Xsample.shape[0]-20, sell_all)
print(f"total profit: {profit:.2f}")
dqn.generate_buy_sell_graph(symbol, iterations, data, profit, 100)
