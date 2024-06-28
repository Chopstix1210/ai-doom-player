from vizdoom import *
import random
import time 
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import cv2
import os 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

# Create a Vizdoom OpenAI Gym environment 
class VizDoomGym(Env):
    def __init__(self, render=False):
        # setup game
        super().__init__
        self.game = DoomGame()
        self.game.load_config('github/ViZDoom/scenarios/basic.cfg')

        # render?
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.init()

        #create action and observation space 
        self.observation_space = Box(low=0, high=255, shape=(3, 240,320), dtype=np.uint8)
        self.action_space = Discrete(3)
    def step(self, action):

        # specify actions and taking step
        actions = np.identity(3, dtype=np.int8).tolist()
        reward = self.game.make_action(actions[action], 4)

        # get game state and info
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = {"ammo": ammo}
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        done = self.game.is_episode_finished()

        return state, reward, done, info
    def close(self):
        self.game.close()
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        return gray
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer 
        return self.grayscale(state)
"""
env = VizDoomGym(True)
print(env_checker.check_env(env))
"""
# Set up Callback 
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(model_path)

        return True
    


CHECKPOINT_DIR = "./train/train_basic"
LOG_DIR = "./logs/log_basic"

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Train the model
env = VizDoomGym()
model = PPO("CnnPolicy", env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)

model.learn(total_timesteps=100000, callback=callback)
