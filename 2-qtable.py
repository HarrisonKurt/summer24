import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from helpers.video import Video
from helpers.stats import Stats



map_name = '8x8'
def run():
    stats = Stats()
    env = gym.make("FrozenLake-v1", map_name = map_name, is_slippery = False) #render_mode = "rgb_array")
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    max_episodes = 100000
    total_rewards = 0
    avg_rewards=[]
    learning_rate = .9
    epsilon = 1.0
    epsilon_decay = 1.0/max_episodes
   
    for episode in range(max_episodes):
        
        terminated = False
        truncated = False
        state =env.reset()[0]

        last_total_rewards = total_rewards
        moves_in_episode = 0

        while(not terminated and not truncated):
            action = env.action_space.sample()
            if np.random.rand() > epsilon:
                action = np.argmax(q_table[state])
            new_state,reward,terminated,truncated,_ = env.step(action)
            total_rewards += reward
            avg_rewards.append(total_rewards/(episode+1))
            q_table[state,action] = (1.0-learning_rate)*q_table[state,action]+learning_rate*.9*(reward+np.max(q_table[new_state]))
            state = new_state
            moves_in_episode += 1

        stats.next_episode(total_rewards, total_rewards-last_total_rewards, moves_in_episode, epsilon, None, 100)
        epsilon = epsilon - epsilon_decay
            
    env.close()
    stats.generate_stats_report("qtable-8x8")

    env = gym.make("FrozenLake-v1", map_name = map_name, is_slippery = False, render_mode = "rgb_array")
    env = Video.create_video_environment(env, "qtable-8x8")
    state = env.reset()[0]
    done = False
    while not done:
        action = np.argmax(q_table[state])
        new_state,reward,terminated,truncated,_ = env.step(action)
        done = terminated or truncated
        state = new_state
    Video.close_video_environment(env)

if __name__ == '__main__':
    run()
    
