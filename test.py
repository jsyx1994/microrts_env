import gym
import rts_wrapper
import os
from rts_wrapper.datatypes import *


# env = gym.make("Microrts-v0")
env = gym.make("CurriculumBaseWorker-v0")
env.action_space.sample()

# example
for _ in range(env.config.max_episodes):
    _, _, done, info = env.reset()

    while not done:
        # action = env.sample(info["unit_valid_actions"])
        action = env.network_simulate(info["unit_valid_actions"])
        _, _, done, info = env.step(action)

    winner = env.get_winner()
    print(winner)

env.close()

# print(rts_wrapper.base_dir_path)

# print(os.path.join(rts_wrapper.base_dir_path, 'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'))
