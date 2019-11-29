import torch
import torch.nn as nn
import torch.nn.functional as F
from rts_wrapper.datatypes import *
from algo.config import model_path
import os
import numpy as np
from torch import Tensor
from rts_wrapper.datatypes import Unit
from rts_wrapper.envs.utils import utt_encoder, unit_feature_encoder
from torch.distributions import Categorical

encoded_utt_dict, encoded_utt_feature_size = utt_encoder(UTT_ORI)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NNBase(nn.Module):
    def __init__(self, out_size):
        super(NNBase, self).__init__()

    def common_func1(self):
        pass

    def common_func2(self):
        pass


class CNNBase(nn.Module):
    def __init__(self, map_height, map_width, input_channel, hidden_size=256, out_size=128, scalar_feature_size=1):
        super(CNNBase, self).__init__()
        # self.map_height = map_height
        # self.map_width = map_width
        self.out_size = out_size
        self.conv_component = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=2), nn.ReLU(),
            nn.Conv2d(64, 128, 2), nn.ReLU(),
            nn.Conv2d(128, 64, 2), nn.ReLU(),
            nn.AdaptiveMaxPool2d((map_height, map_width)),  # n * 64 * map_height * map_width
            # self-attention?
            Flatten()
        )

        self.linear_component = nn.Sequential(
            nn.Linear(64 * map_height * map_width + scalar_feature_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, out_size), nn.ReLU()
        )
        # try:
        #     self.load_state_dict(torch.load(os.path.join(model_path, 'base.pt')))
        # except FileNotFoundError as e:
        #     print(e)
        #     torch.save(self.state_dict(), os.path.join(model_path, 'base.pt'))

    def forward(self, spatial_feature, scalar_feature):
        x = spatial_feature
        x = self.conv_component(x)
        x = torch.cat([x, scalar_feature], dim=1)
        x = self.linear_component(x)
        return x


class Critic(nn.Module):
    def __init__(self, base: CNNBase = None):
        super(Critic, self).__init__()

        self.shared = base

        self.mlp_component = nn.Sequential(
            nn.Linear(in_features=base.out_size, out_features=64), nn.ReLU(),
            nn.Linear(in_features=64, out_features=32), nn.ReLU(),
        )
        self.critic_linear = nn.Linear(32, 1)

    def forward(self, spatial_feature, scalar_feature):
        x = self.shared(spatial_feature, scalar_feature)
        inner_out = x
        x = self.mlp_component(x)
        x = self.critic_linear(x)
        return x, inner_out

    def get_value_evaluated(self, spatial_feature, scalar_feature):
        x = self.shared(spatial_feature, scalar_feature)
        x = self.mlp_component(x)
        x = self.critic_linear(x)
        return x

    def get_inner_flow(self, spatial_feature, scalar_feature):
        x = self.shared(spatial_feature, scalar_feature)
        return x


class Actor(nn.Module):
    def __init__(self, base_out_size, unit_feature_size=18, scalar_feature_size=2):
        super(Actor, self).__init__()
        self.activated_agents = [
            # UNIT_TYPE_NAME_BASE,
            # UNIT_TYPE_NAME_BARRACKS,
            UNIT_TYPE_NAME_WORKER,
            # UNIT_TYPE_NAME_HEAVY,
            # UNIT_TYPE_NAME_LIGHT,
            # UNIT_TYPE_NAME_RANGED,
        ]
        self.base_out_layer = nn.Sequential(
            nn.Linear(base_out_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.utt_embedding_layer = nn.Sequential(
            nn.Linear(encoded_utt_feature_size + unit_feature_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        self.pre_gru_layer = nn.Sequential(
            nn.Linear(128 + 128 + scalar_feature_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        # gru coding here

        self.actors = nn.ModuleDict({
            UNIT_TYPE_NAME_WORKER: nn.Sequential(
                # nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, WorkerAction.__members__.items().__len__()),  # logits
                nn.Softmax(dim=1)
            ),
            UNIT_TYPE_NAME_BASE: nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, BaseAction.__members__.items().__len__()),
                nn.Softmax(dim=1),
            )

        })

    def deterministic_action_sampler(self, actor_type: str, base_out: torch.Tensor, unit_feature: torch.Tensor,
                                     scalar_feature: torch.Tensor):
        """
        :param actor_type:
        :param base_out:
        :param unit_feature:
        :param scalar_feature:
        :return: Unit action type from datatypes
        """
        if actor_type not in self.activated_agents:
            return AGENT_ACTIONS_MAP[actor_type].DO_NONE

        probs = self.forward(actor_type, base_out, unit_feature, scalar_feature)
        # print(prob)
        return list(AGENT_ACTIONS_MAP[actor_type])[torch.argmax(probs).item()]

    def stochastic_action_sampler(self, actor_type: str, base_out: torch.Tensor, unit_feature: torch.Tensor,
                                  scalar_feature: torch.Tensor):
        if actor_type not in self.activated_agents:
            return AGENT_ACTIONS_MAP[actor_type].DO_NONE

        probs = self.forward(actor_type, base_out, unit_feature, scalar_feature)
        m = Categorical(probs)
        idx = m.sample().item()
        action = list(AGENT_ACTIONS_MAP[actor_type])[idx]
        return action

    def forward(self, actor_type: str, base_out: torch.Tensor, unit_feature: torch.Tensor,
                scalar_feature: torch.Tensor):
        encoded_utt = torch.from_numpy(encoded_utt_dict[actor_type]).float().unsqueeze(0)
        # encoded_unit = torch.from_numpy(unit_feature).float().unsqueeze(0)

        x1 = self.base_out_layer(base_out)
        x2 = self.utt_embedding_layer(torch.cat([encoded_utt, unit_feature], dim=1))
        x3 = scalar_feature
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.pre_gru_layer(x)

        output = self.actors[actor_type](x)
        return output


class ActorCritic(nn.Module):

    def __init__(self, map_height, map_width, input_channel=21, unit_feature_size=18):
        super(ActorCritic, self).__init__()
        self.shared_out_size = 256

        self.activated_agents = [
            # UNIT_TYPE_NAME_BASE,
            # UNIT_TYPE_NAME_BARRACKS,
            UNIT_TYPE_NAME_WORKER,
            # UNIT_TYPE_NAME_HEAVY,
            # UNIT_TYPE_NAME_LIGHT,
            # UNIT_TYPE_NAME_RANGED,
        ]

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=2), nn.ReLU(),
            nn.Conv2d(64, 128, 2), nn.ReLU(),
            nn.Conv2d(128, 64, 2), nn.ReLU(),
            nn.AdaptiveMaxPool2d((map_height, map_width)),  # n * 64 * map_height * map_width
            # self-attention?
            Flatten()
        )

        self.shared_linear = nn.Sequential(
            nn.Linear(64 * map_height * map_width, 512), nn.ReLU(),
            nn.Linear(512, self.shared_out_size), nn.ReLU(),
        )

        self.critic_mlps = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.critic_out = nn.Linear(32, 1)

        self.actor_mlps = nn.Sequential(
            nn.Linear(self.shared_out_size + unit_feature_size + encoded_utt_feature_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )

        self.actor_out = nn.ModuleDict({
            UNIT_TYPE_NAME_WORKER: nn.Sequential(
                # nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, WorkerAction.__members__.items().__len__()),  # logits
                nn.Softmax(dim=1)
            ),
            UNIT_TYPE_NAME_BASE: nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, BaseAction.__members__.items().__len__()),
                nn.Softmax(dim=1),
            )
        })

    def _shared_forward(self, spatial_feature):
        x = self.shared_conv(spatial_feature)
        x = self.shared_linear(x)
        return x

    def evaluate(self, spatial_feature: Tensor):
        self.eval()
        x = self._shared_forward(spatial_feature)
        x = self.critic_mlps(x)
        x = self.critic_out(x)
        return x

    def critic_forward(self, spatial_feature: Tensor):
        x = self._shared_forward(spatial_feature)
        x = self.critic_mlps(x)
        x = self.critic_out(x)
        return x

    def actor_forward(self, actor_type: str, spatial_feature: Tensor, unit_feature: Tensor):
        encoded_utt = torch.from_numpy(encoded_utt_dict[actor_type]).float().unsqueeze(0)
        x = self._shared_forward(spatial_feature)
        x = torch.cat([x, encoded_utt, unit_feature], dim=1)
        x = self.actor_mlps(x)
        probs = self.actor_out[actor_type](x)
        return probs

    def deterministic_action_sampler(self, actor_type: str, spatial_feature: Tensor, unit_feature: Tensor):
        if actor_type not in self.activated_agents:
            return AGENT_ACTIONS_MAP[actor_type].DO_NONE

        probs = self.actor_forward(actor_type, spatial_feature, unit_feature)
        # print(prob)
        return list(AGENT_ACTIONS_MAP[actor_type])[torch.argmax(probs).item()]

    def stochastic_action_sampler(self, actor_type: str, spatial_feature: Tensor, unit_feature: Tensor):
        if actor_type not in self.activated_agents:
            return AGENT_ACTIONS_MAP[actor_type].DO_NONE

        probs = self.actor_forward(actor_type, spatial_feature, unit_feature)
        m = Categorical(probs)
        idx = m.sample().item()
        action = list(AGENT_ACTIONS_MAP[actor_type])[idx]
        return action


def gradient_for_inner_connection_out_of_cnnbase_test():
    """test result gradient works"""
    from torch import optim
    torch.manual_seed(1)
    input_data = torch.randn(1, 18, 6, 6)
    base = CNNBase(6, 6, 18)
    critic = Critic(base)

    value, cnnout = critic(input_data)
    print(value)

    actor = Actor(cnnout.size(1), "Worker")

    actor_out = actor(cnnout)
    print(actor_out)
    loss = nn.MSELoss()(actor_out, torch.Tensor([[100]]))
    print(loss)

    actor.zero_grad()
    loss.backward()

    params = list(actor.parameters()) + list(critic.parameters())
    # print(type(params))
    optimizer = optim.SGD(params, lr=.1)
    optimizer.step()

    print(critic(input_data)[0])
    print(actor(cnnout))


def test_network():
    import json
    unit_entity_str = '{"type":"Worker", "ID":22, "player":0, "x":0, "y":2, "resources":0, "hitpoints":1}'
    pgs_wrapper_str = '{"reward":140.0,"done":false,"validActions":[{"unit":{"type":"Worker", "ID":22, "player":0, "x":0, "y":2, "resources":0, "hitpoints":1},"unitActions":[{"type":1, "parameter":1} ,{"type":1, "parameter":2} ,{"type":0, "parameter":10}]},{"unit":{"type":"Worker", "ID":24, "player":0, "x":3, "y":4, "resources":0, "hitpoints":1},"unitActions":[{"type":1, "parameter":0} ,{"type":1, "parameter":1} ,{"type":1, "parameter":2} ,{"type":1, "parameter":3} ,{"type":0, "parameter":10}]}],"gs":{"time":164,"pgs":{"width":6,"height":6,"terrain":"000000000000000000000000000000000000","players":[{"ID":0, "resources":2},{"ID":1, "resources":5}],"units":[{"type":"Resource", "ID":0, "player":-1, "x":0, "y":0, "resources":230, "hitpoints":1},{"type":"Base", "ID":19, "player":1, "x":5, "y":5, "resources":0, "hitpoints":10},{"type":"Base", "ID":20, "player":0, "x":2, "y":2, "resources":0, "hitpoints":10},{"type":"Worker", "ID":22, "player":0, "x":0, "y":2, "resources":0, "hitpoints":1},{"type":"Worker", "ID":23, "player":0, "x":5, "y":2, "resources":0, "hitpoints":1},{"type":"Worker", "ID":24, "player":0, "x":3, "y":4, "resources":0, "hitpoints":1},{"type":"Worker", "ID":25, "player":0, "x":0, "y":1, "resources":0, "hitpoints":1},{"type":"Worker", "ID":26, "player":0, "x":2, "y":3, "resources":0, "hitpoints":1}]},"actions":[{"ID":20, "time":153, "action":{"type":4, "parameter":1, "unitType":"Worker"}},{"ID":26, "time":158, "action":{"type":1, "parameter":3}},{"ID":19, "time":160, "action":{"type":0, "parameter":10}},{"ID":25, "time":162, "action":{"type":2, "parameter":0}},{"ID":23, "time":163, "action":{"type":1, "parameter":2}}]}}'
    unit = from_dict(data_class=Unit, data=json.loads(unit_entity_str))
    gs_wrapper = from_dict(data_class=GsWrapper, data=json.loads(pgs_wrapper_str))

    scalar_feature_actor = np.array([p.resources for p in gs_wrapper.gs.pgs.players])
    rsrc1, rsrc2 = scalar_feature_actor
    scalar_feature_critic = np.array([0.5 if rsrc1 == rsrc2 else (rsrc1 - rsrc2) / (rsrc1 + rsrc2)])
    unit_feature = unit_feature_encoder(unit, gs_wrapper.gs.pgs.height, gs_wrapper.gs.pgs.width)

    # print(scalar_feature)

    cnnbase = CNNBase(6, 6, 19)
    input_data = torch.randn(1, 19, 6, 6)

    critic = Critic(cnnbase)

    base_out = cnnbase(input_data, torch.from_numpy(scalar_feature_critic).unsqueeze(0).float())

    actor = Actor(base_out.size(1))
    print(actor("Worker", base_out, unit_feature, scalar_feature_actor))
    # print(actor)
    # print(actor(base_out,))
    # actor = Actor()np.ndarray
    # print(actor)


if __name__ == '__main__':
    import json

    unit_entity_str = '{"type":"Worker", "ID":22, "player":0, "x":0, "y":2, "resources":0, "hitpoints":1}'
    pgs_wrapper_str = '{"reward":140.0,"done":false,"validActions":[{"unit":{"type":"Worker", "ID":22, "player":0, "x":0, "y":2, "resources":0, "hitpoints":1},"unitActions":[{"type":1, "parameter":1} ,{"type":1, "parameter":2} ,{"type":0, "parameter":10}]},{"unit":{"type":"Worker", "ID":24, "player":0, "x":3, "y":4, "resources":0, "hitpoints":1},"unitActions":[{"type":1, "parameter":0} ,{"type":1, "parameter":1} ,{"type":1, "parameter":2} ,{"type":1, "parameter":3} ,{"type":0, "parameter":10}]}],"gs":{"time":164,"pgs":{"width":6,"height":6,"terrain":"000000000000000000000000000000000000","players":[{"ID":0, "resources":2},{"ID":1, "resources":5}],"units":[{"type":"Resource", "ID":0, "player":-1, "x":0, "y":0, "resources":230, "hitpoints":1},{"type":"Base", "ID":19, "player":1, "x":5, "y":5, "resources":0, "hitpoints":10},{"type":"Base", "ID":20, "player":0, "x":2, "y":2, "resources":0, "hitpoints":10},{"type":"Worker", "ID":22, "player":0, "x":0, "y":2, "resources":0, "hitpoints":1},{"type":"Worker", "ID":23, "player":0, "x":5, "y":2, "resources":0, "hitpoints":1},{"type":"Worker", "ID":24, "player":0, "x":3, "y":4, "resources":0, "hitpoints":1},{"type":"Worker", "ID":25, "player":0, "x":0, "y":1, "resources":0, "hitpoints":1},{"type":"Worker", "ID":26, "player":0, "x":2, "y":3, "resources":0, "hitpoints":1}]},"actions":[{"ID":20, "time":153, "action":{"type":4, "parameter":1, "unitType":"Worker"}},{"ID":26, "time":158, "action":{"type":1, "parameter":3}},{"ID":19, "time":160, "action":{"type":0, "parameter":10}},{"ID":25, "time":162, "action":{"type":2, "parameter":0}},{"ID":23, "time":163, "action":{"type":1, "parameter":2}}]}}'
    unit = from_dict(data_class=Unit, data=json.loads(unit_entity_str))
    gs_wrapper = from_dict(data_class=GsWrapper, data=json.loads(pgs_wrapper_str))

    scalar_feature = torch.randn(1,2)
    # rsrc1, rsrc2 = scalar_feature_actor
    # scalar_feature_critic = np.array([0.5 if rsrc1 == rsrc2 else (rsrc1 - rsrc2) / (rsrc1 + rsrc2)])
    unit_feature = torch.from_numpy(unit_feature_encoder(unit, gs_wrapper.gs.pgs.height, gs_wrapper.gs.pgs.width)).float().unsqueeze(0)
    input_data = torch.randn(1, 21, 6, 6)

    model = ActorCritic(4, 4)
    print(model.evaluate(input_data))  # critic test
    print(model.stochastic_action_sampler(unit.type, input_data, unit_feature))
