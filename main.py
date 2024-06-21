from vizdoom import *
import random
import time 
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import cv2

# Create a Vizdoom OpenAI Gym environment 
class VizDoomGym(Env):
    def __init__(self, render=False):
        # setup game
        game = DoomGame()
        game.load_config('github/ViZDoom/scenarios/basic.cfg')

        #create action and observation space 
        self.observation_space = Box(low=0, high=255, shape=(3, 240,320), dtype=np.uint8)
        self.action_space = Discrete(3)


        # render?
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        game.init()
    def step(self, action):

        # specify actions and taking step
        actions = np.identity(3, dtype=np.int8).tolist()
        reward = self.game.make_action(actions[action], 4)

        # get game state and info
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            info = state.game_variables
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0
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

env = VizDoomGym(True)
