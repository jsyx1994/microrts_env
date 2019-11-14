import gym
import os
from gym import Space, spaces
from subprocess import Popen, PIPE
import socket
from rts_wrapper.utils.socket_utils import get_available_port
import json
from dacite import from_dict
from .space import DictSpace
import numpy as np
from rts_wrapper.datatypes import *

action_collection = [BaseAction, BarracksAction, WorkerActon, LightAction, HeavyAction]


class MicroRts(gym.Env):
    config = None
    port = None
    conn = None
    server_socket = None

    def __init__(self, config=''):
        np.random.seed()
        self.random = np.random
        self.unit_action_map = {}
        for action in action_collection:
            self.unit_action_map[action.__type_name__] = action

        self.action_space = DictSpace({
            'Base': spaces.Discrete(BaseAction.__members__.items().__len__()),
            'Barracks': spaces.Discrete(BarracksAction.__members__.items().__len__()),
            'Worker': spaces.Discrete(WorkerActon.__members__.items().__len__()),
            'Light': spaces.Discrete(LightAction.__members__.items().__len__()),
            'Heavy': spaces.Discrete(HeavyAction.__members__.items().__len__()),
            'Ranged': spaces.Discrete(RangedAction.__members__.items().__len__()),

        })

        self.config = config
        if config:
            self.init_server()
            # comments the next line then in developer mode
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
        gs_wrapper = from_dict(data_class=GsWrapper, data=json.loads(raw.split('\n')[1]))
        observation = self.parse_game_state(gs_wrapper.gs, curr_player)
        reward = gs_wrapper.reward
        done = gs_wrapper.done
        info = {
            "unit_valid_actions": gs_wrapper.validActions
        }
        return observation, reward, done, info

    def network_simulate(self, unit_valid_actions: List[UnitValidAction]):
        """
        :param unit_valid_actions:
        :return: choice for every unit, adding UVA to check validation later
        """
        unit_validaction_choices = []
        for uva in unit_valid_actions:
            choice = self.random.choice(list(self.unit_action_map[uva.unit.type]))
            # (choice) BaseAction.DO_NONE.name, BaseAction.DO_NONE.value
            unit_validaction_choices.append((uva, choice))
            print(choice)
        print(self.network_action_translator(unit_validaction_choices))

    def network_action_translator(self, unit_validaction_choices) -> List[PlayerAction]:
        pas = []
        for uva, choice in unit_validaction_choices:
            unit = uva.unit
            valid_actions = uva.unitActions

            # valid_probe_directions = [va.parameter for va in valid_actions if
            #                           va.type != ACTION_TYPE_PRODUCE and va.parameter in (
            #                               ACTION_PARAMETER_DIRECTION_UP,
            #                               ACTION_PARAMETER_DIRECTION_RIGHT,
            #                               ACTION_PARAMETER_DIRECTION_DOWN,
            #                               ACTION_PARAMETER_DIRECTION_LEFT
            #                           )
            #                           ]
            # valid_produce_directions = [va.parameter for va in valid_actions if ]

            def get_valid_probe_action(direction) -> UnitAction:
                ua = [action for action in valid_actions if
                      action.parameter == direction.value and action.type != ACTION_TYPE_PRODUCE]

                assert len(ua) == 1
                return ua[0] if ua else ua

            def get_valid_produce_directions(unit_type) -> List:
                return [action.parameter for action in valid_actions if action.unitType == unit_type]

            pa = PlayerAction(unitID=unit.ID)
            if unit.type == UNIT_TYPE_NAME_BASE:
                if choice == BaseAction.DO_NONE:
                    pa.unitAction.type = ACTION_TYPE_NONE

                elif choice == BaseAction.DO_LAY_WORKER:
                    directions = get_valid_produce_directions(UNIT_TYPE_NAME_WORKER)
                    if not directions:
                        print("Invalid network action, do nothing")
                        pass
                    else:
                        pa.unitAction.type = ACTION_TYPE_PRODUCE
                        pa.unitAction.parameter = self.random.choice(directions)

            elif unit.type == UNIT_TYPE_NAME_BARRACKS:
                if choice == BarracksAction.DO_NONE:
                    pa.unitAction.type = ACTION_TYPE_NONE

                elif choice == BarracksAction.DO_LAY_LIGHT:
                    pa.unitAction.type = ACTION_TYPE_PRODUCE
                    pa.unitAction.parameter = self.random.choice(get_valid_produce_directions(UNIT_TYPE_NAME_LIGHT))

                    pa.unitAction.unitType = UNIT_TYPE_NAME_LIGHT

                elif choice == BarracksAction.DO_LAY_HEAVY:
                    pa.unitAction.type = ACTION_TYPE_PRODUCE
                    pa.unitAction.parameter = self.random.choice(get_valid_produce_directions(UNIT_TYPE_NAME_HEAVY))

                    pa.unitAction.unitType = UNIT_TYPE_NAME_HEAVY

                elif choice == BarracksAction.DO_LAY_RANGED:
                    pa.unitAction.type = ACTION_TYPE_PRODUCE
                    pa.unitAction.parameter = self.random.choice(get_valid_produce_directions(UNIT_TYPE_NAME_RANGED))

                    pa.unitAction.unitType = UNIT_TYPE_NAME_RANGED

            elif unit.type == UNIT_TYPE_NAME_WORKER:
                if choice == WorkerActon.DO_NONE:
                    pa.unitAction.type = ACTION_TYPE_NONE

                elif choice == WorkerActon.DO_UP_PROBE:
                    pa.unitAction = get_valid_probe_action(WorkerActon.DO_UP_PROBE)

                elif choice == WorkerActon.DO_RIGHT_PROBE:
                    pa.unitAction = get_valid_probe_action(WorkerActon.DO_RIGHT_PROBE)

                elif choice == WorkerActon.DO_DOWN_PROBE:
                    pa.unitAction = get_valid_probe_action(WorkerActon.DO_DOWN_PROBE)

                elif choice == WorkerActon.DO_LEFT_PROBE:
                    pa.unitAction = get_valid_probe_action(WorkerActon.DO_LEFT_PROBE)

                elif choice == WorkerActon.DO_LAY_BASE:
                    pa.unitAction = self.random.choice(get_valid_produce_directions(UNIT_TYPE_NAME_BASE))

                elif choice == WorkerActon.DO_LAY_BARRACKS:
                    pa.unitAction = self.random.choice(get_valid_produce_directions(UNIT_TYPE_NAME_BARRACKS))

            elif unit.type == UNIT_TYPE_NAME_LIGHT:
                if choice == LightAction.DO_NONE:
                    pa.unitAction.type = ACTION_TYPE_NONE

                elif choice == LightAction.DO_UP_PROBE:
                    pa.unitAction = get_valid_probe_action(LightAction.DO_UP_PROBE)

                elif choice == LightAction.DO_RIGHT_PROBE:
                    pa.unitAction = get_valid_probe_action(LightAction.DO_RIGHT_PROBE)

                elif choice == LightAction.DO_DOWN_PROBE:
                    pa.unitAction = get_valid_probe_action(LightAction.DO_DOWN_PROBE)

                elif choice == LightAction.DO_LEFT_PROBE:
                    pa.unitAction = get_valid_probe_action(LightAction.DO_LEFT_PROBE)

            elif unit.type == UNIT_TYPE_NAME_HEAVY:
                if choice == HeavyAction.DO_NONE:
                    pa.unitAction.type = ACTION_TYPE_NONE

                elif choice == HeavyAction.DO_UP_PROBE:
                    pa.unitAction = get_valid_probe_action(HeavyAction.DO_UP_PROBE)

                elif choice == HeavyAction.DO_RIGHT_PROBE:
                    pa.unitAction = get_valid_probe_action(HeavyAction.DO_RIGHT_PROBE)

                elif choice == HeavyAction.DO_DOWN_PROBE:
                    pa.unitAction = get_valid_probe_action(HeavyAction.DO_DOWN_PROBE)

                elif choice == HeavyAction.DO_LEFT_PROBE:
                    pa.unitAction = get_valid_probe_action(HeavyAction.DO_LEFT_PROBE)

            elif unit.type == UNIT_TYPE_NAME_RANGED:
                pass

            pas.append(pa)
        return pas

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
        print(raw)
        return self.signal_wrapper(raw)

    def render(self, mode='human'):
        pass

    @staticmethod
    def sample(unit_valid_actions: List[UnitValidAction]) -> List[PlayerAction]:
        pas = []
        import random
        for uas in unit_valid_actions:
            x = random.choice(uas.unitActions)
            pas.append(
                asdict(PlayerAction(
                    unitID=uas.unit.ID,
                    unitAction=x
                ))
            )
        print(json.dumps(pas))
        return pas

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
