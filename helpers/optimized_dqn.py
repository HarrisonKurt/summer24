# https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
# https://github.com/MarcinPerka/FrozenLakeRL/tree/main/frozenlake

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
from helpers.stats import Stats
from helpers.video import Video
import copy

class OptimizedDQN:
  def __init__(self, map_name):
    self.map_name = map_name
    self.env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=False)

    input_features = self.env.observation_space.n
    hidden_features_l1 = 150
    hidden_features_l2 = 75
    output_features = self.env.action_space.n

    self.q_network = nn.Sequential(
      nn.Linear(input_features, hidden_features_l1),
      nn.ReLU(),
      nn.Linear(hidden_features_l1, hidden_features_l2),
      nn.ReLU(),
      nn.Linear(hidden_features_l2, output_features),
    )

    self.target_network = nn.Sequential(
      nn.Linear(input_features, hidden_features_l1),
      nn.ReLU(),
      nn.Linear(hidden_features_l1, hidden_features_l2),
      nn.ReLU(),
      nn.Linear(hidden_features_l2, output_features),
    )

    self.target_network = copy.deepcopy(self.q_network)
    self.target_network.load_state_dict(self.q_network.state_dict())

    self.memory = deque(maxlen=2500)

  def epsilon_greedy(self, q_values, epsilon):
    if np.random.rand() < epsilon:
      action = self.env.action_space.sample()
    else:
      action = q_values.argmax().item()
    return action


  def train(self, training_episodes):
    stats = Stats()
    optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    sync_frequency = 50
    batch_size = 2000
    total_steps = 0
    total_reward = 0
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = (epsilon - min_epsilon) / training_episodes

    last_total_reward = 0

    for episode in range(training_episodes):
      s = self.env.reset()[0]
      state = torch.zeros(self.env.observation_space.n)
      state[s] = 1.0

      episode_loss = 0
      moves_in_episode = 0
      done = False
      
      while not done:
        moves_in_episode += 1
        total_steps += 1

        if moves_in_episode == 1:
          q_values = self.q_network(state)
          action = self.epsilon_greedy(q_values, epsilon)

        ns, reward, terminated, truncated, _ = self.env.step(action)
        total_reward += reward
        done = terminated or truncated
          
        next_state = torch.zeros(self.env.observation_space.n)
        next_state[ns] = 1.0

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
        # instead of directly training the model on each iteration, we are going to save a bunch
        # of moves in to a memory buffer. We will then create a minibatch and train the model
        # on that batch to try and match the target q values - as the buffer gets filled with
        # "more correct" moves, the training will converge quicker

        self.memory.append(())

        q_values = next_q_values
        state = next_state
        action = next_action

      if len(self.memory) > batch_size:
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

        # TODO
        # let's discuss the calculation of the target_q value
        # this is really just looks more complex because we are dealing with a batch instead of a single value
        #
        # next_action_batch.unsqueeze(1) - this just adds a dimension to the next_action batch
        # for example if it was [1, 2, 3, 4] before this, after it would be [[1], [2], [3], [4]]
        #
        # next_state_target_Q_values.gather(1, next_action_batch.unsqueeze(1)) - this takes the target q value for each action above
        # think of this as pulling out the item from the target_q_value array based on the action array
        # for example if the next_next_state_target_Q_values was:
        # [
        #   [1.0, 0.0, 0.0, 0.0],
        #   [0.0, 2.0, 0.0, 0.0],
        #   [0.0, 0.0, 3.0, 0.0],
        #   [0.0, 0.0, 0.0, 4.0],
        # ]
        # and the unsqueezed next action batch was: [[0], [1], [2], [3]]
        # the result of this would be [[1.0], [2.0], [3.0], [4.0]] since it would pull out the value at the index for each row
        #
        # next_state_target_Q_values.gather(1, next_action_batch.unsqueeze(1)).squeeze() - this just removes that extra dimension
        # so if before the squeeze the values were: [[1.0], [2.0], [3.0], [4.0]], after they would be [1.0, 2.0, 3.0, 4.0]
        #
        # (1.0 - done_batch.int()) * next_state_target_Q_values.gather(1, next_action_batch.unsqueeze(1)).squeeze() - this just makes q values
        # set to zero for anything that was in the done state, for example if the done vector is: [0, 0, 1, 0], then 1-done = [1, 1, 0, 1]
        # and multiplying that vector by the q_value vector will turn everything that IS DONE to 0.
        #
        # the last step just adds the rewards to these target q values, but does it for the whole batch at once
        target_q = reward_batch + 0.9 * (1.0 - done_batch.int()) * next_state_target_Q_values.gather(1, next_action_batch.unsqueeze(1)).squeeze()

        # TODO
        # get the q values for the actual_q, you will need something similar to the code above
        # what you want is the Q_values for the actions that were taken.
        actual_q = None

        # calculate loss
        loss = F.mse_loss(actual_q, target_q, reduction='mean')
        episode_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      epsilon -= epsilon_decay
      if epsilon < min_epsilon:
          epsilon = min_epsilon

      stats.next_episode(total_reward, (total_reward-last_total_reward), moves_in_episode, None, episode_loss / batch_size, 100)
      last_total_reward = total_reward

      if episode % sync_frequency == 0:
          self.target_network.load_state_dict(self.q_network.state_dict())

    # one last copy to target network before ending
    self.target_network.load_state_dict(self.q_network.state_dict())

    self.env.close()

    stats.generate_stats_report(f"optimized-dqn-{self.map_name}-{training_episodes}")

    if not os.path.exists("./models"):
      os.makedirs("./models")
    torch.save(self.q_network.state_dict(), f"./models/optimized-dqn-{self.map_name}-{training_episodes}.mdl")

  def run(self, training_episodes):
    self.q_network.load_state_dict(torch.load(f"./models/optimized-dqn-{self.map_name}-{training_episodes}.mdl"))

    env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=False, render_mode="rgb_array")
    env = Video.create_video_environment(env, f"optimized-dqn-{self.map_name}-{training_episodes}")

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
