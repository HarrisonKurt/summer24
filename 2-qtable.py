import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt



map_name = '8x8'
def run():
    env = gym.make("FrozenLake-v1", map_name = map_name, is_slippery = False) #render_mode = "human")
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    max_episodes = 10000
    total_rewards = 0
    avg_rewards=[]
    learning_rate = .9
    epsilon = 1.0
    epsilon_decay = 1.0/max_episodes
    print(epsilon_decay)
   
    for episode in range(max_episodes):
        
        terminated = False
        truncated = False
        state =env.reset()[0]
        while(not terminated and not truncated):
            action = env.action_space.sample()
            if np.random.rand() > epsilon:
                action = np.argmax(q_table[state])
            new_state,reward,terminated,truncated,_ = env.step(action)
            total_rewards += reward
            avg_rewards.append(total_rewards/(episode+1))
            q_table[state,action] = (1.0-learning_rate)*q_table[state,action]+learning_rate*.9*(reward+np.max(q_table[new_state]))
            state = new_state
            #if reward > 0:
        epsilon = epsilon - epsilon_decay
            
    env.close()
    print(avg_rewards)
    #plt.plot(avg_rewards)
    #plt.show()
    print(q_table)
    
    

    env = gym.make("FrozenLake-v1", map_name = map_name, is_slippery = False, render_mode = "human")
    state = env.reset()[0]
    done = False
    while not done:
        action = np.argmax(q_table[state])
        new_state,reward,terminated,truncated,_ = env.step(action)
        done = terminated or truncated
        state = new_state




    

if __name__ == '__main__':
    run()
    
