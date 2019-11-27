import gym
import rts_wrapper
import os
from rts_wrapper.datatypes import *
from rts_wrapper.envs.utils import *
import torch

# env = gym.make("Microrts-v0")
# env = gym.make("CurriculumBaseWorker-v0")
env = gym.make("OneWorkerAndResources-v0")

# env.action_space.sample()

from algo.model import CNNBase, Critic, Actor


def initialize_network(height, width, channel):
    cnnbase = CNNBase(height, width, channel)
    critic_net = Critic(cnnbase)
    actor_net = Actor(cnnbase.out_size)
    return actor_net, critic_net


def sample(actor_net: Actor, critic_net: Critic, state, info):
    units = [uva.unit for uva in info["unit_valid_actions"]]
    height, width = info["map_size"]
    player_resources = info["player_resources"]
    current_player = info["current_player"]

    base_out = critic_net.get_inner_flow(state)
    for u in units:
        unit_feature = unit_feature_encoder(u, height, width)
    pass


actor, critic = initialize_network(env.config.height, env.config.width, 19)
# example
for _ in range(env.config.max_episodes):
    state, reward, done, info = env.reset()

    while not done:
        # action = env.sample(info["unit_valid_actions"])

        action = env.network_simulate(info["unit_valid_actions"])
        state, reward, done, info = env.step(action)

    winner = env.get_winner()   # required
    print(winner)

env.close()

# print(rts_wrapper.base_dir_path)

# print(os.path.join(rts_wrapper.base_dir_path, 'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'))
