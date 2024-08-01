import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", map_name = "4x4", is_slippery = False) #render_mode = "human")
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def run():
    max_episodes = 500
    total_rewards = 0
    avg_rewards=[]
    learning_rate = .9
    epsilon = .8
   
    for episode in range(max_episodes):
        
        terminated = False
        truncated = False
        state =env.reset()[0]
        while(not terminated and not truncated):
            action = env.action_space.sample()
            print(epsilon)
            if np.random.rand() > epsilon:
                action = np.argmax(q_table[state])
            new_state,reward,terminated,truncated,_ = env.step(action)
            total_rewards += reward
            avg_rewards.append(total_rewards/(episode+1))
            q_table[state,action] = (1.0-learning_rate)*q_table[state,action]+learning_rate*(reward+np.max(q_table[new_state]))
            state = new_state
            if reward > 0:
                epsilon = epsilon - .1
    env.close()
    print(avg_rewards)
    plt.plot(avg_rewards)
    plt.show()

if __name__ == '__main__':
    run()
    print(q_table)
