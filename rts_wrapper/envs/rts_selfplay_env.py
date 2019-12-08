from rts_wrapper import Config
from rts_wrapper.envs import MicroRts


class SelfPlayEnv(MicroRts):
    def __init__(self, config: Config):
        super().__init__(config)