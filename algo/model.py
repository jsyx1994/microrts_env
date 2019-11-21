import torch
import torch.nn as nn
import torch.nn.functional as F
from rts_wrapper.datatypes import AGENT_COLLECTION, UTT_DICT


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NNBase(nn.Module):
    def __init__(self):
        super(NNBase, self).__init__()

    def common_func1(self):
        pass

    def common_func2(self):
        pass


class CNNBase(nn.Module):
    def __init__(self, map_height, map_width, input_channel, hidden_size=512, out_size=256):
        super(CNNBase, self).__init__()
        # self.map_height = map_height
        # self.map_width = map_width
        self.conv_component = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=2), nn.ReLU(),
            nn.Conv2d(64, 128, 2), nn.ReLU(),
            nn.Conv2d(128, 64, 2), nn.ReLU(),
            nn.AdaptiveMaxPool2d((map_height, map_width)),   # n * 64 * map_height * map_width
            # self-attention?
            Flatten()
        )

        self.linear_component = nn.Sequential(
            nn.Linear(64 * map_height * map_width, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, out_size), nn.ReLU()
        )
        # try:
            # self.load_state_dict()

    def forward(self, input):
        x = input
        x = self.conv_component(x)
        x = self.linear_component(x)
        return x


class Critic(nn.Module):
    def __init__(self, map_height, map_width):
        super(Critic, self).__init__()
        hidden_size, out_size = 256, 128
        self.shared = CNNBase(map_height, map_width, 18, hidden_size, out_size)

        self.mlp_component = nn.Sequential(
            nn.Linear(in_features=out_size, out_features=64), nn.ReLU(),
            nn.Linear(in_features=64, out_features=32), nn.ReLU(),
        )
        self.critic_linear = nn.Linear(32, 1)

    def forward(self, input):
        x = input
        x = self.shared(x)
        x = self.mlp_component(x)
        x = self.critic_linear(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self):
        pass


class Actor(nn.Module):
    def __init__(self, actor_type):
        super(Actor, self).__init__()
        assert actor_type in AGENT_COLLECTION


if __name__ == '__main__':
    print(UTT_DICT)
    inx = torch.randn(1, 18, 8, 8)
    # model = CNNBase(6, 6, 18)
    # print(model)
    # print(model(inx).size())
    # print(model(inx))
    model = Critic(6, 6)
    print(model)
    print(model(inx).size())
    print(model(inx))
    pass



