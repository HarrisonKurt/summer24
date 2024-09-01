import matplotlib.pyplot as plt
import numpy as np
import os

class Stats:
  def __init__(self):
    self.cumulative_rewards = []
    self.episode_rewards = []
    self.episode_moves = []
    self.epsilons = []
    self.losses = []
    self.validation_rewards = []

    self.episode = 0

  def next_episode(self, cumulative_reward, episode_reward, moves, epsilon, loss, print_every):
    if cumulative_reward != None:
      self.cumulative_rewards.append(cumulative_reward)
    if episode_reward != None:
      self.episode_rewards.append(episode_reward)
    if moves != None:
      self.episode_moves.append(moves)
    if epsilon != None:
      self.epsilons.append(epsilon)
    if loss != None:
      self.losses.append(loss)

    self.episode += 1

    if self.episode % print_every == 0:
      print(f"episode: {self.episode}, epsilon: {epsilon}, cumulative_reward: {cumulative_reward}, avg episode rewards: {np.average(self.episode_rewards[-print_every:])}, avg moves: {np.average(self.episode_moves[-print_every:])}, avg loss: {np.average(self.losses[-100:])}, validation reward: {np.average(self.validation_rewards[-print_every:])}")

  def next_validation_reward(self, validation_reward, print_every):
    self.validation_rewards.append(validation_reward)

  def generate_stats_report(self, name):
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    if len(self.losses) > 0:
      axs[0, 0].set_title('loss')
      axs[0, 0].plot(self.losses)
    elif len(self.epsilons) > 0:
      axs[0, 0].set_title('epsilon')
      axs[0, 0].plot(self.epsilons)

    if len(self.cumulative_rewards) > 0:
      axs[1, 0].set_title('cumulative reward')
      axs[1, 0].plot(self.cumulative_rewards)

    if len(self.episode_rewards) > 0:
      axs[0, 1].set_title('episode rewards')
      axs[0, 1].plot(self.episode_rewards)
    elif len(self.epsilons) > 0:
      axs[0, 1].set_title('epsilon')
      axs[0, 1].plot(self.epsilons)
    
    if len(self.episode_moves) > 0:
      axs[1, 1].set_title('moves per episode')
      axs[1, 1].plot(self.episode_moves)
    elif len(self.validation_rewards) > 0:
      axs[1, 1].set_title('validation reward')
      axs[1, 1].plot(self.validation_rewards)

    fig.tight_layout(pad=2.0)

    if not os.path.exists("./stats"):
      os.makedirs("./stats")
    plt.savefig(f"./stats/{name}.png")
