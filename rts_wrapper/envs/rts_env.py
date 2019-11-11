import gym
import os
from gym import Space, spaces
from subprocess import Popen, PIPE
import socket
from rts_wrapper.utils.socket_utils import get_available_port
import json
from rts_wrapper.datatypes import *
from dacite import from_dict
from .space import DictSpace
import numpy as np


class MicroRts(gym.Env):
    config = None
    port = None
    conn = None
    server_socket = None

    def __init__(self, config=''):
        self.action_space = DictSpace({
            'a': spaces.Discrete(10),
            'b': spaces.Discrete(3)
        })

        self.config = config
        if config:
            self.init_server()
            # self.init_client()
            print(config)
            print(MicroRts.reward_range)
            self.establish_connection()

    def init_client(self):
        """
        before-interacting setting-ups
        """
        setup_commands = [
            "java",
            "-jar",
            os.path.join(os.path.expanduser('~/microrts_env/rts_wrapper'),
                         'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'),
            "--port", str(self.port),
            "--map", os.path.join(os.path.expanduser(self.config.microrts_path), self.config.map_path),
            "more",
            "options"
        ]
        java_client = Popen(
            setup_commands,
            stdin=PIPE,
            stdout=PIPE
        )
        stdout, stderr = java_client.communicate()
        print(stdout.decode("utf-8"))

    def init_server(self):
        self.port = get_available_port()
        self.server_socket = socket.socket()
        self.server_socket.bind((self.config.client_ip, 9898))

    def establish_connection(self):
        self.server_socket.listen(5)
        print("Wait for Java client connection...")
        self.conn, address_info = self.server_socket.accept()
        print("Server: Send welcome msg to client...")
        print(self._send_msg("Welcome meg sent!"))

    def _send_msg(self, msg: str):
        try:
            self.conn.send(('%s\n' % msg).encode('utf-8'))
        except Exception as err:
            print("An error has occured: ", err)
        return self.conn.recv(65536).decode('utf-8')

    def signal_wrapper(self, raw):
        curr_player = int(raw.split('\n')[0].split()[1])
        print(curr_player)
        gs_wrapper = from_dict(data_class=GsWrapper, data=json.loads(raw.split('\n')[1]))
        observation = self.parse_game_state(gs_wrapper.gs, curr_player)
        reward = None
        done = gs_wrapper.done
        info = {
            "unit_valid_actions": gs_wrapper.validActions
        }
        return observation, reward, done, info

    def step(self, action: List[Any]):
        """
        :param action: '{"unitID": "", "unitAction":{"type":"", "parameter": -1, "x":-1,"y":-1, "unitType":""}}'
        :return: observation, reward, done, info[List[Unit]]
        info:{
            "unit_valid_actions":
        }
        """
        # print(self._send_msg('[]'))

        raw = self._send_msg(json.dumps(action))
        return self.signal_wrapper(raw)

    def reset(self):
        print("Server: Send reset command...")
        raw = self._send_msg('reset')
        return self.signal_wrapper(raw)

    def render(self, mode='human'):
        pass

    @staticmethod
    def sample(unit_valid_actions: List[UnitValidAction]) -> List[PlayerAction]:
        pa = []
        import random
        for uas in unit_valid_actions:
            x = random.choice(uas.unitActions)
            pa.append(
                asdict(PlayerAction(
                    unitID=uas.unit.ID,
                    unitAction=x
                ))
            )
        print(json.dumps(pa))
        return pa

    @staticmethod
    def parse_game_state(gs: GameState, player):
        current_player = player

        # Used for type indexing
        utt = ['Base', 'Barracks', 'Worker', 'Light', 'Heavy', 'Ranged']
        type_idx = {}
        for i, ut in zip(range(len(utt)), utt):
            type_idx[ut] = i

        time = gs.time
        pgs = gs.pgs
        actions = gs.actions
        w = pgs.width
        h = pgs.height
        units = pgs.units

        # Initialization of spatial features
        spatial_features = np.zeros((18, h, w))

        # channel_wall
        spatial_features[0] = np.array([int(x) for x in pgs.terrain]).reshape((1, h, w))

        # other channels
        channel_resource = spatial_features[1]
        channel_self_type = spatial_features[2:8]
        channel_self_hp = spatial_features[8]
        channel_self_resource_carried = spatial_features[9]
        channel_enemy_type = spatial_features[10:16]
        channel_enemy_hp = spatial_features[16]
        channel_enemy_resource_carried = spatial_features[17]

        for unit in units:
            _player = unit.player
            _type = unit.type
            x = unit.x
            y = unit.y
            # neutral
            if _player == -1:
                channel_resource[x][y] = unit.resources
                # channel_resource[x][y] = 1

            elif _player == current_player:
                # get the index of this type
                idx = type_idx[_type]
                channel_self_type[idx][x][y] = 1
                channel_self_hp[x][y] = unit.hitpoints
                channel_self_resource_carried[x][y] = unit.resources

            else:
                idx = type_idx[_type]
                channel_enemy_type[idx][x][y] = 1
                channel_enemy_hp[x][y] = unit.hitpoints
                channel_enemy_resource_carried[x][y] = unit.resources
        # print(spatial_features)
        # print(spatial_features.shape)
        return spatial_features

    def close(self):
        self.conn.close()
