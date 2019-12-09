from typing import List
import os
from subprocess import PIPE, Popen
from gym import spaces
import gym

from rts_wrapper.datatypes import Config, UnitValidAction, AGENT_ACTIONS_MAP, BaseAction, BarracksAction, WorkerAction, \
    LightAction, HeavyAction, RangedAction
from rts_wrapper.envs.utils import network_action_translator, get_available_port
from rts_wrapper.envs.player import Player


class BaseEnv(gym.Env):
    setup_commands = None
    config = None
    # players = []

    def __init__(self, config: Config):
        """
        initialize, do not call any other member function, leaving it to the successor
        :param config:
        """
        self.config = config
        self._init_client()
        # self._counting_players()
        self.action_space = ({
            'Base': spaces.Discrete(BaseAction.__members__.items().__len__()),
            'Barracks': spaces.Discrete(BarracksAction.__members__.items().__len__()),
            'Worker': spaces.Discrete(WorkerAction.__members__.items().__len__()),
            'Light': spaces.Discrete(LightAction.__members__.items().__len__()),
            'Heavy': spaces.Discrete(HeavyAction.__members__.items().__len__()),
            'Ranged': spaces.Discrete(RangedAction.__members__.items().__len__()),
        })

    @property
    def map_size(self):
        return self.config.height, self.config.width

    @property
    def max_episodes(self):
        return self.config.max_episodes

    @property
    def ai1_type(self):
        return self.config.ai1_type

    @property
    def ai2_type(self):
        return self.config.ai2_type

    def _add_commands(self, option, args):
        assert isinstance(option, str), "Option should be string"
        assert option.startswith("--"), "Invalid option"
        assert args is not None, "Args should be filled"

        self.setup_commands.append(option)
        self.setup_commands.append(args)

    def _init_client(self):
        """
        before-interacting setting-ups, and open the java program. Need to add port in kids
        """
        self.setup_commands = [
            "java",
            "-jar",
            os.path.join(os.path.expanduser('~/microrts_env/rts_wrapper'),
                         'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'),
            "--map", os.path.join(os.path.expanduser(self.config.microrts_path), self.config.map_path),
            "--ai1_type", self.config.ai1_type,
            "--ai2_type", self.config.ai2_type,
            "--maxCycles", str(self.config.max_cycles),
            "--maxEpisodes", str(self.config.max_episodes),
            "--period", str(self.config.period),
            "--render", str(self.config.render),
            # "--port", str(self.port),
            # "more",
            # "options"
        ]

    def start_client(self):
        print(' '.join(self.setup_commands))
        java_client = Popen(
            self.setup_commands,
            stdin=PIPE,
            stdout=PIPE
        )
        stdout, stderr = java_client.communicate()
        print(stdout.decode("utf-8"))
        pass

    def network_simulate(self, unit_valid_actions: List[UnitValidAction]):
        """
        This　imitate the network outputs for testing
        :param unit_valid_actions: the units' valid actions from gs wrapper
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

    def step(self, action):
        raise NotImplementedError

    def reset(self, **kwargs):
        """
        all players say hello to server
        """
        raise NotImplementedError

    def render(self, mode='human'):
        pass


