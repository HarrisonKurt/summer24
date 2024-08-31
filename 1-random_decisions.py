import gymnasium as gym
from helpers.video import Video

def run():
    env = gym.make("FrozenLake-v1", map_name = "4x4", is_slippery = False, render_mode = "rgb_array")
    env = Video.create_video_environment(env, "random-4x4")

    state =env.reset()
    terminated = False
    truncated = False
    
    while(not terminated and not truncated):
        action = env.action_space.sample()
        new_state,reward,terminated,truncated,_ = env.step(action)
        state = new_state

    Video.close_video_environment(env)

if __name__ == '__main__':
    run()
