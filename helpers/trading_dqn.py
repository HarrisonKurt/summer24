# https://github.com/pskrunner14/trading-bot/tree/master

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import drange
import matplotlib.dates as mdates

from helpers.data import Data
from helpers.stack import Stack
from helpers.stats import Stats

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import copy
from collections import deque
import os

class TradingDQN:
  def __init__(self, lookback_days, buy_reward, sell_penalty, hold_reward):
    self.lookback_days = lookback_days
    self.buy_reward = buy_reward
    self.sell_penalty = sell_penalty
    self.hold_reward = hold_reward

    self.output_features = 3
    hidden_features_l1 = 128
    hidden_features_l2 = 256
    hidden_features_l3 = 128

    self.q_network = nn.Sequential(
      nn.Linear(self.lookback_days, hidden_features_l1),
      nn.ReLU(),
      nn.Linear(hidden_features_l1, hidden_features_l2),
      nn.ReLU(),
      nn.Linear(hidden_features_l2, hidden_features_l3),
      nn.ReLU(),
      nn.Linear(hidden_features_l3, self.output_features),
    )

    self.target_network = nn.Sequential(
      nn.Linear(self.lookback_days, hidden_features_l1),
      nn.ReLU(),
      nn.Linear(hidden_features_l1, hidden_features_l2),
      nn.ReLU(),
      nn.Linear(hidden_features_l2, hidden_features_l3),
      nn.ReLU(),
      nn.Linear(hidden_features_l3, self.output_features),
    )

    self.target_network = copy.deepcopy(self.q_network)
    self.target_network.load_state_dict(self.q_network.state_dict())

    self.memory = deque(maxlen=2500)

  def epsilon_greedy(self, q_values, epsilon):
    if np.random.rand() < epsilon:
      action = np.random.randint(0, self.output_features)
    else:
      action = q_values.argmax().item()
    return action

  def calculate_reward(self, current_price, action, holdings, include_incentives = True, sell_all = False):
    # TODO: calculate the reward based on the current action
    # current_price - the price of the asset today
    # action - action to take, either 0 = buy, 1 = sell, 2 = hold
    # holdings - a stack of the prices you have purchased the asset in the past, if holdings is empty, you don't current hold any
    # sell_all - if when you sell, you sell all your holdings at the current price, or just the lastest - defaults to False (just sell last)
    #
    # we also have buy_reward, sell_penalty, and hold_reward to try and incentivize certain behaviors.
    # e.g., a high sell reward, makes selling less profitable and will train the model to sell less frequently
    # only include these if include_incentives is set to True
    reward = 0
    if action == 0:
      # add the latest price to your holdings
      holdings.push(current_price)
      if include_incentives:
        reward = self.buy_reward
    elif action == 1 and holdings.len() > 0:
      profit = 0

      if sell_all:
        while holdings.len() > 0:
          buy_price = holdings.pop()
          profit += (current_price - buy_price)
      else:
        buy_price = holdings.pop()
        profit = (current_price - buy_price)
            
      # reward is equal to the profit from the transaction (this could be negative)
      reward = profit
      if include_incentives:
        reward -= self.sell_penalty
    else:
      if holdings.len() > 0 and include_incentives:
        reward = self.hold_reward

    return reward

  def calculate_total_profit(self, data, network, start_day, length, generate_buy_sell_data, sell_all = False):
    if generate_buy_sell_data:
      data['action'] = ''
      data['profit'] = 0.0

    stack = Stack()

    total_profit = 0
    day = start_day

    while day < length:
      prices, state = Data.get_state(data, day, self.lookback_days)
      current_price = prices[self.lookback_days - 1].item()

      q_values = network(state)    
      action = q_values.argmax().item()

      stack_length = stack.len()
      profit = self.calculate_reward(current_price, action, stack, False, sell_all)

      if generate_buy_sell_data:
        if action == 0:
          data.loc[day, 'action'] = 'B'
        elif action == 1 and stack_length > 0:
          data.loc[day, 'action'] = 'S'
          data.loc[day, 'profit'] = profit

      total_profit += profit
      day += 1

    return total_profit, data
  
  def train(self, symbol, Xtrain, Xval, training_episodes, sell_all = False):
    stats = Stats()
    optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = (epsilon - min_epsilon) / training_episodes
    batch_size = 2000
    sync_frequency = 50

    total_steps = 0

    for episode in range(training_episodes):
      holdings = Stack()
      episode_profit = 0

      # pick a random start day
      initial_day = np.random.randint(self.lookback_days + 1, 200)
      day = initial_day

      while day < Xtrain.shape[0]:
        price, state = Data.get_state(Xtrain, day, self.lookback_days)
  
        # default reward is 0
        reward = 0
        if day == initial_day:
          action = 0 # always buy on the first day
        else:
          action = self.epsilon_greedy(self.q_network(state), epsilon)
  
        price, next_state = Data.get_state(Xtrain, day+1, self.lookback_days)
        next_action = self.target_network(next_state).argmax().item()
  
        # if we have nothing to sell, then just hold
        if action == 1 and holdings.len() == 0:
          action = 2

        current_price = price[self.lookback_days - 1].item()
        reward = self.calculate_reward(current_price, action, holdings, sell_all)
        episode_profit += reward
  
        done = (day == Xtrain.shape[0] - 1)
        self.memory.append((state, action, next_state, reward, done, next_action))
  
        state = next_state
        action = next_action
  
        day += 1        

      if len(self.memory) > batch_size:
        total_steps += 1

        # create the minibatch
        ix = torch.randint(0, len(self.memory)-1, (batch_size, ))
        state_batch = torch.stack([(self.memory[i][0]) for i in ix])
        action_batch = torch.tensor([self.memory[i][1] for i in ix])
        next_state_batch = torch.stack([self.memory[i][2] for i in ix])
        reward_batch = torch.tensor([self.memory[i][3] for i in ix])
        done_batch = torch.tensor([self.memory[i][4] for i in ix])
        next_action_batch = torch.tensor([self.memory[i][5] for i in ix])

        # get the current and future Q values
        Q_values = self.q_network(state_batch)
        with torch.no_grad():
          next_state_target_Q_values = self.target_network(next_state_batch)

        target_q = reward_batch + 0.9 * (1.0 - done_batch.int()) * next_state_target_Q_values.gather(1, next_action_batch.unsqueeze(1)).squeeze()
        actual_q = Q_values.gather(1, action_batch.unsqueeze(1)).squeeze()

        # calculate loss
        loss = F.mse_loss(actual_q, target_q, reduction='mean')
        episode_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epsilon -= epsilon_decay
        if epsilon < min_epsilon:
            epsilon = min_epsilon

        stats.next_episode(episode_profit, None, None, epsilon, episode_loss / batch_size, 100)

        if episode % sync_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            val_profit, _ = self.calculate_total_profit(Xval, self.target_network,  self.lookback_days + 1, Xval.shape[0], False, sell_all)
            stats.next_validation_reward(val_profit, sync_frequency)

    # one last copy to target network before ending
    self.target_network.load_state_dict(self.q_network.state_dict())
    stats.generate_stats_report(f"trading-dqn-{symbol}-{training_episodes}")

    if not os.path.exists("./models"):
      os.makedirs("./models")
    torch.save(self.q_network.state_dict(), f"./models/trading-dqn-{symbol}-{training_episodes}.mdl")

  def run(self, symbol, training_episodes, data, length, sell_all = False):
    self.q_network.load_state_dict(torch.load(f"./models/trading-dqn-{symbol}-{training_episodes}.mdl"))

    experiment_length = length
    start_day = data.shape[0] - experiment_length

    return self.calculate_total_profit(data, self.q_network, start_day, experiment_length, True, sell_all)

  def generate_buy_sell_graph(self, symbol, training_episodes, data, profit, days = 50):
    data = data[-days:].reset_index()
    
    buys = data.loc[data['action']=='B'].index
    sells = data.loc[data['action']=='S'].index
    
    figure, axis = plt.subplots(figsize=(12, 8), dpi=200)

    months = mdates.MonthLocator()

    plt.title(f"Total Profit: ${profit:.2f}")
    plt.plot('Date', 'Adj Close', data = data)
    axis.xaxis.set_major_locator(months)
    axis.grid(True)
    
    plt.gcf().autofmt_xdate()
    
    for tb in buys:
      axis.annotate(data.iloc[tb]['action'], (tb, data.iloc[tb]['Adj Close'] - 1.5))
    for ts in sells:
      axis.annotate(data.iloc[ts]['action'], (ts, data.iloc[ts]['Adj Close'] + 1.5))

    figure.tight_layout(pad=2.0)

    name = f"trading-dqn-{symbol}-{training_episodes}-buysell"
    if not os.path.exists("./stats"):
      os.makedirs("./stats")
    plt.savefig(f"./stats/{name}.png")


