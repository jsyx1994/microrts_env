import gym
import os
from gym import spaces
from subprocess import Popen, PIPE
import socket
from rts_wrapper.envs.utils import *
from .space import DictSpace
import numpy as np
from rts_wrapper.datatypes import *
from threading import Thread



class MicroRts(gym.Env):
    config = None
    port = None
    conn = None
    game_time = None
    DEBUG = 0

    def __init__(self, config: Config):
        self.config = config

        self.random = rd
        # self.unit_action_map = {}
        self.port = get_available_port()
        # self.encoded_utt_dict = utt_encoder(UTT_ORI)
        # print(self.encoded_utt_dict)
        # print(config.utt)
        # for action in action_collection:
        #     self.unit_action_map[action.__type_name__] = action

        self.action_space = DictSpace({
            'Base': spaces.Discrete(BaseAction.__members__.items().__len__()),
            'Barracks': spaces.Discrete(BarracksAction.__members__.items().__len__()),
            'Worker': spaces.Discrete(WorkerAction.__members__.items().__len__()),
            'Light': spaces.Discrete(LightAction.__members__.items().__len__()),
            'Heavy': spaces.Discrete(HeavyAction.__members__.items().__len__()),
            'Ranged': spaces.Discrete(RangedAction.__members__.items().__len__()),

        })

        # start connection
        Thread(target=self.init_client).start()

        self.init_server()

        # comments the next line then in developer mode
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
            "--ai1_type", self.config.ai1_type,
            "--ai2_type", self.config.ai2_type,
            "--maxCycles", str(self.config.max_cycles),
            "--maxEpisodes", str(self.config.max_episodes),
            # "more",
            # "options"
        ]
        print(' '.join(setup_commands))
        java_client = Popen(
            setup_commands,
            stdin=PIPE,
            stdout=PIPE
        )
        stdout, stderr = java_client.communicate()
        print(stdout.decode("utf-8"))

    def init_server(self):
        server_socket = socket.socket()
        server_socket.bind((self.config.client_ip, self.port))
        server_socket.listen(5)
        print("Wait for Java client connection...")
        self.conn, address_info = server_socket.accept()

    def establish_connection(self):
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
        print(raw)
        input()
        observation = state_encoder(gs_wrapper.gs, curr_player)
        reward = gs_wrapper.reward
        done = gs_wrapper.done
        self.game_time = gs_wrapper.gs.time
        info = {
            "unit_valid_actions": gs_wrapper.validActions,  # friends and their valid actions
            # "units_list": [u for u in gs_wrapper.gs.pgs.units],
            # "enemies_actions": None,
            "current_player": curr_player,
            "player_resources": [p.resources for p in gs_wrapper.gs.pgs.players],
            "map_size": [gs_wrapper.gs.pgs.height, gs_wrapper.gs.pgs.width],
            "time_stamp": gs_wrapper.gs.time,
        }
        return observation, reward, done, info

    def network_simulate(self, unit_valid_actions: List[UnitValidAction]):
        """
        :param unit_valid_actions:
        :return: choice for every unit, adding UVA to check validation later
        """
        unit_validaction_choices = []
        if self.game_time % (self.config.frame_skip + 1) == 0:
            for uva in unit_valid_actions:
                # if uva.unit.type == "Worker":
                #     choice = WorkerAction.DO_LAY_BARRACKS
                # else:
                choice = self.random.choice(list(AGENT_ACTIONS_MAP[uva.unit.type]))
                # (choice) BaseAction.DO_NONE.name, BaseAction.DO_NONE.value
                unit_validaction_choices.append((uva, choice))
                if self.DEBUG:
                    print(choice)
            return network_action_translator(unit_validaction_choices)
        else:
            print(self.game_time)
            print("skip")
            return unit_validaction_choices

    def step(self, action: List[PlayerAction]):
        """
        :param action: '{"unitID": "", "unitAction":{"type":"", "parameter": -1, "x":-1,"y":-1, "unitType":""}}'
        :return: observation, reward, done, info[List[Unit]]
        info:{
            "unit_valid_actions":
        }
        """
        # print(self._send_msg('[]'))
        pa = pa_to_jsonable(action)
        raw = self._send_msg(pa)
        return self.signal_wrapper(raw)

    def reset(self):
        print("Server: Send reset command...")
        raw = self._send_msg('reset')
        # print(raw)
        return self.signal_wrapper(raw)

    def render(self, mode='human'):
        pass

    def get_winner(self):
        """
        this function must come after the done is true (i.e. game is over)
        -1: tie
        0: player 0
        1: player 1
        :return:
        """
        return self.conn.recv(1024).decode('utf-8')

    def sample(self, unit_valid_actions: List[UnitValidAction]) -> List[PlayerAction]:
        pas = []
        import random
        for uas in unit_valid_actions:
            x = random.choice(uas.unitActions)
            pas.append(
                PlayerAction(
                    unitID=uas.unit.ID,
                    unitAction=x
                )
            )
        # print(json.dumps(pas))
        return pas

    def close(self):
        self.conn.close()
