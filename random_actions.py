from vizdoom import *
import random
import time 
import numpy as np

#setup game 
game = DoomGame()
game.load_config('github/ViZDoom/scenarios/basic.cfg')
game.init()

# setting to create random actions 
actions = np.identity(3, dtype=np.uint8).tolist()

episodes = 10
for episode in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        info = state.game_variables
        reward = game.make_action(random.choice(actions))
        print('reward: ', reward)
        time.sleep(0.02)
    print('Result: ', game.get_total_reward())
    time.sleep(2)
