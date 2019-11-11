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
        return self.conn.recv(4096).decode('utf-8')

    def step(self, action):
        """
        :param action: '{"unitID": "", "unitAction":{"type":"", "parameter": -1, "x":-1,"y":-1, "unitType":""}}'
        :return: observation, reward, done, info[List[Unit]]
        """
        print(self._send_msg('[]'))
        pass

    def reset(self):
        print("Server: Send reset command...")
        raw = self._send_msg('reset')
        print(raw)
        gs_wrapper = from_dict(data_class=GsWrapper, data=json.loads(raw.split('\n')[1]))
        unit_actions = list(gs_wrapper.validActions)
        return None, None, False, {"unit_valid_actions": unit_actions}

    def render(self, mode='human'):
        pass

    @staticmethod
    def sample(unit_valid_actions: List[UnitValidAction]):
        pa = []
        import random
        for uas in unit_valid_actions:
            x = random.choice(uas.unitActions)
            pa.append(asdict(PlayerAction(
                unitID=uas.unit.ID,
                unitAction=x
            )))
        print(json.dumps(pa))
        return pa

