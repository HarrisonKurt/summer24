import numpy as np
import gymnasium as gym

class Video:
  def create_video_environment(env, type):
      video_env = gym.wrappers.RecordVideo(env=env, video_folder='./video', name_prefix=type, episode_trigger=lambda x: True)
      return video_env

  def close_video_environment(video_env):
      video_env.close_video_recorder()
      video_env.close()

    