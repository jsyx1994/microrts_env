import gym
import os
from subprocess import Popen, PIPE
from rts_wrapper.utils.socket_utils import get_available_port
from rts_wrapper import base_dir_path

class MicroRts(gym.Env):

    def __init__(self, config=''):
        """
        setting the client and server socket
        """
        self.config = config
        self.java_client = None
        self.port = ''

        if config:
            self.init_server()
            self.init_client()
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
        self.java_client = Popen(
            setup_commands,
            stdin=PIPE,
            stdout=PIPE
        )

    def init_server(self):
        self.port = get_available_port()

    def establish_connection(self):
        stdout, stderr = self.java_client.communicate()
        print(stdout.decode("utf-8"))
        # print(stderr)


    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
