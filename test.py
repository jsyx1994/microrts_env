import gym
import rts_wrapper
import os
from rts_wrapper import datatypes

env = gym.make("Microrts-v0")

_, _, _, info = env.reset()
print(env.sample(info["unit_valid_actions"]))

for _ in range(10000):
    env.step(action=env.action_space.sample())

env.action_space.sample()
print(rts_wrapper.base_dir_path)
print(os.path.join(rts_wrapper.base_dir_path, 'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'))
