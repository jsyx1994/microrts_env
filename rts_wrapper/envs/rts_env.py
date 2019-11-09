import gym
import os
from subprocess import Popen, PIPE
import socket
from rts_wrapper.utils.socket_utils import get_available_port
from rts_wrapper import base_dir_path

class MicroRts(gym.Env):
    config = None
    port = None
    conn = None
    server_socket = None

    def __init__(self, config=''):
        """
        setting the client and server socket
        """
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
            os.path.join(os.path.expanduser('~/microrts_env/rts_wrapper'), 'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'),
            "--port", str(self.port),
            "--map",  os.path.join(os.path.expanduser(self.config.microrts_path), self.config.map_path),
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
        print("Waiting Java client connection...")
        self.conn, address_info = self.server_socket.accept()
        print(self._send_msg('hello there'))
        print(self._send_msg('hello there'))
        print(self._send_msg('hello there'))
        # print(stderr)

    def _send_msg(self, msg: str):
        try:
            self.conn.send(('%s\n' % msg).encode('utf-8'))
        except Exception as err:
            print("An error has occured: ", err)
        return self.conn.recv(4096).decode('utf-8')


    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
