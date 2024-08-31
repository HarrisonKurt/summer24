import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
from helpers.stats import Stats
from helpers.video import Video

class SimpleDQN:
  def __init__(self, map_name):
    self.map_name = map_name
    self.env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=False)

    input_features = self.env.observation_space.n
    hidden_features_l1 = 150
    hidden_features_l2 = 75
    output_features = self.env.action_space.n

    # TODO
    # build the self.q_network neural network
    # it will be a squential model that will have inputs for every state, e.g., if there were 4 possible states it would
    # take a "one hot" vector of 4 values that describes the current state - for state 1 it would look like: [0, 1, 0, 0]
    # it's output will be the action space, e.g., if there are 4 possible outputs then it would output
    # a "one hot" vector of 4 values that descries the action to take - for action 0 it would look like [1, 0, 0, 0]
    # it will have two hidden layers
    # we will use ReLU activation function between the layers

    self.q_network = nn.Sequential(
      nn.Linear(input_features, hidden_features_l1),
      nn.ReLU(),
      nn.Linear(hidden_features_l1, hidden_features_l2),
      nn.ReLU(),
      nn.Linear(hidden_features_l2, output_features),
    )

  def epsilon_greedy(self, q_values, epsilon):
    # TODO
    # just like in q_table
    # pick a random number, if it's less than epsilon pick a random action
    # if it's greater than epsilon then pick the action that corresponds to the largest item in q_values (e.g., the index of the largest value)

    if np.random.rand() < epsilon:
        action = self.env.action_space.sample()
    else:
        action = q_values.argmax().item()
    return action

  def train(self, training_episodes):
    optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    total_steps = 0
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = (epsilon - min_epsilon)/training_episodes

    stats = Stats()

    total_reward = 0
    last_total_reward = 0

    
    for episode in range(training_episodes):
      s = self.env.reset()[0]
      state = torch.zeros(1, self.env.observation_space.n)
      state[0, s] = 1.0

      moves_in_episode = 0
      episode_loss = 0
    
      done = False
      while not done:
        moves_in_episode += 1
        total_steps += 1

        # TODO: 
        # use the q_network and epsilon_greedy to select a next action
        # perform the action
        # use a one-hot vector to build the next state and calculate the next action

        q_values = self.q_network(state)
        action = self.epsilon_greedy(q_values, epsilon)

        ns, reward, truncated, terminated, _ = self.env.step(action)
        done = truncated or terminated
        total_reward += reward

        next_state = torch.zeros(1, self.env.observation_space.n)
        next_state[0, ns] = 1.0

        next_q_values = self.q_network(next_state)
        next_action = self.epsilon_greedy(next_q_values, epsilon)

        adjusted_reward = reward
        if reward > 0.0 and done:
          # reached the goal state
          adjusted_reward = 10.0
        elif done:
          # didn't reach the goal state, either exceeded max moves or fell in a hole
          adjusted_rewared = -10.0
        else:
          # moved to a safe square, but not the goal state
          adjusted_rewared = -1.0

        # TODO    
        # update the target q-value
        # this will be very similar to the q_table calculation of target, although we've removed learning_rate
        # the target will be the reward we just calculated plus the maximum q value for the next move
        # however, if we are "done" we shouldn't add the next move q value because we've already adjusted
        # the rewards and penalties above

        max_next_q_value = next_q_values.max()        
        target = adjusted_reward + (1-done) * max_next_q_value
        target_q = torch.zeros(1, self.env.action_space.n)
        target_q[:] = q_values[:]
        target_q[0, action] = target

        # calculate the loss
        loss = F.mse_loss(q_values, target_q, reduction='mean')
        episode_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        action = next_action

      stats.next_episode(total_reward, (total_reward-last_total_reward), moves_in_episode, None, episode_loss/moves_in_episode, 100)

      epsilon -= epsilon_decay
      last_total_reward = total_reward

    self.env.close()

    stats.generate_stats_report(f"simple-dqn-{self.map_name}-{training_episodes}")

    if not os.path.exists("./models"):
      os.makedirs("./models")
    torch.save(self.q_network.state_dict(), f"./models/simple-dqn-{self.map_name}-{training_episodes}.mdl")

  def run(self, training_episodes):
    self.q_network.load_state_dict(torch.load(f"./models/simple-dqn-{self.map_name}-{training_episodes}.mdl"))

    env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=False, render_mode="rgb_array")
    env = Video.create_video_environment(env, f"simple-dqn-{self.map_name}-{training_episodes}")

    s = env.reset()[0]
    state = torch.zeros(env.observation_space.n)
    state[s] = 1.0

    actions = []
    done = False
    while not done:
      q_values = self.q_network(state)
      action = q_values.argmax().item()
      actions.append(action)

      ns, reward, truncated, terminated, _ = env.step(action)

      state = torch.zeros(env.observation_space.n)
      state[ns] = 1.0

      done = truncated or terminated

    Video.close_video_environment()