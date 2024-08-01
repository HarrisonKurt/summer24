import gymnasium as gym
def run():
    env = gym.make("FrozenLake-v1", map_name = "4x4", is_slippery = False, render_mode = "human")

    state =env.reset()
    terminated = False
    truncated = False
    
    while(not terminated and not truncated):
        action = env.action_space.sample()
        new_state,reward,terminated,truncated,_ = env.step(action)
        state = new_state
    env.close()

if __name__ == '__main__':
    run()
