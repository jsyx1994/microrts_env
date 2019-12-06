import gym
import torch
from rts_wrapper.envs.utils import action_sampler_v1


def evaluate_game(eval_env: str, model, mode="stochastic"):
    device = torch.device("cpu")
    model.to(device)
    env = gym.make(eval_env)
    for _ in range(env.config.max_episodes):
        obs_t, _, done, info_t = env.reset()  # deserting the reward
        while not done:
            # action = env.sample(info["unit_valid_actions"])
            # action = env.network_simulate(info_t["unit_valid_actions"])
            # action = action_sampler_v0(actor, critic, state_t, info_t)
            action = action_sampler_v1(model, obs_t, info_t, mode)
            # print(action)
            state_tp1, reward, done, info_tp1 = env.step(action)

            obs_t, info_t = state_tp1, info_tp1   # assign new state and info from environment
            # replay_buffer.push(state_t,info_t,ac)

        winner = env.get_winner()  # required
        print(winner)

    env.close()