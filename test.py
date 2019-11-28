import gym
import rts_wrapper
import os
from rts_wrapper.datatypes import *
from rts_wrapper.envs.utils import *
import torch
from algo.replay_buffer import ReplayBuffer

# env = gym.make("Microrts-v0")
# env = gym.make("CurriculumBaseWorker-v0")
env = gym.make("OneWorkerAndBaseWithResources-v0")

# env.action_space.sample()

from algo.model import CNNBase, Critic, Actor


def initialize_network(height, width, channel):
    cnnbase = CNNBase(height, width, channel)
    critic_net = Critic(cnnbase)
    actor_net = Actor(cnnbase.out_size)
    return actor_net, critic_net


def action_sampler(actor_net: Actor, critic_net: Critic, state, info) -> List[Any]:
    # units = [uva.unit for uva in info["unit_valid_actions"]]
    time_stamp = info["time_stamp"]
    # if time_stamp % 1 != 0:
    #     return []
    unit_valid_actions = info["unit_valid_actions"]     # unit and its valid actions
    height, width = info["map_size"]
    player_resources = info["player_resources"]  # global resource situation, default I'm player 0
    current_player = info["current_player"]

    rsrc_my, rsrc_opp = player_resources
    if current_player == 1:
        rsrc_my, rsrc_opp = rsrc_opp, rsrc_my

    scalar_feature_actor = torch.tensor([rsrc_my, rsrc_opp]).float().unsqueeze(0)
    scalar_feature_critic = torch.tensor([.5]) if rsrc_my == rsrc_opp == 0 else torch.tensor(
        [(rsrc_my - rsrc_opp) / (rsrc_my + rsrc_opp)]).unsqueeze(0)
    spatial_feature = torch.from_numpy(state).float().unsqueeze(0)

    base_out = critic_net.get_inner_flow(spatial_feature, scalar_feature_critic)

    samples = []
    for uva in unit_valid_actions:
        u = uva.unit
        unit_feature = torch.from_numpy(unit_feature_encoder(u, height, width)).float().unsqueeze(0)
        sampled_unit_action = actor_net.stochastic_action_sampler(u.type, base_out, unit_feature, scalar_feature_actor)
        samples.append((uva, sampled_unit_action))

    return network_action_translator(samples)


actor, critic = initialize_network(env.config.height, env.config.width, 19)
# example
replay_buffer = ReplayBuffer(100)
for _ in range(env.config.max_episodes):
    state_t, _, done, info_t = env.reset()  # deserting the reward

    while not done:
        # action = env.sample(info["unit_valid_actions"])
        # action = env.network_simulate(info["unit_valid_actions"])
        action = action_sampler(actor, critic, state_t, info_t)
        print(action)
        state_tp1, reward, done, info_tp1 = env.step(action)




        state_t, info_t = state_tp1, info_tp1   # assign new state and info from environment
        # replay_buffer.push(state_t,info_t,ac)

    winner = env.get_winner()  # required
    print(winner)

env.close()

# print(rts_wrapper.base_dir_path)

# print(os.path.join(rts_wrapper.base_dir_path, 'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'))
