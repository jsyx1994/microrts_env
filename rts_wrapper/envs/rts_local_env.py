from rts_wrapper import Config
from rts_wrapper.envs import MicroRts
import gym
from multiprocessing import Process, Pool
from threading import  Thread
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from rts_wrapper.envs import BaseEnv
from rts_wrapper.envs.player import Player
from rts_wrapper.envs.utils import get_available_port, network_simulator
from rts_wrapper.datatypes import Observations
from dacite import from_dict


class BattleEnv(BaseEnv):
    players = []

    @property
    def players_num(self):
        return len(self.players)

    def __init__(self, config):
        super(BattleEnv, self).__init__(config)
        self.DEBUG = False
        self._counting_players()

        if not self.DEBUG:
            Thread(target=self.start_client).start()
        self._players_join()

    def _counting_players(self):
        if self.ai1_type.startswith("socket"):
            player = Player(0, self.config.client_ip, 9898 if self.DEBUG else get_available_port())
            # player = Player(0, self.config.client_ip, get_available_port())
            self._add_commands("--port" + str(1 + player.id), str(player.port))
            self.players.append(player)
        if self.ai2_type.startswith("socket"):
            player = Player(1, self.config.client_ip, 9898 if self.DEBUG else get_available_port())

            # if self.DEBUG:
            #     player = Player(1, self.config.client_ip, 8787)
            # else:
            #     player = Player(1, self.config.client_ip, get_available_port())
            self._add_commands("--port" + str(1 + player.id), str(player.port))
            self.players.append(player)

    def _players_join(self):
        for player in self.players:
            player.join()

    @staticmethod
    def _obs2dataclass(obs):
        return Observations(
                    observation=obs[0],
                    reward=obs[1],
                    done=obs[2],
                    info=obs[3]
            )

    # TODO: get
    def step(self, action: list):
        """
        local env should using Threads to avoid dead lock, while remote needn't
        """
        import time
        for i in range(self.players_num):
            self.players[i].act(action[i])
            # time.sleep(.1)

        signals_res = [self._obs2dataclass(player.observe()) for player in self.players]
        return signals_res

    def reset(self, **kwargs):
        signals = [self._obs2dataclass(player.reset()) for player in self.players]
        return signals

    def get_winner(self):
        """
        this function must come after the done is true (i.e. game is over)
        -1: tie
        0: player 0
        1: player 1
        :return:
        """
        for p in self.players:
            p.expect()


def main():
    env = gym.make("LargeMapTest-v0")
    players = env.players

    for _ in range(env.max_episodes):
        obses = env.reset()  # p1 and p2 reset
        while not obses[0].done:
            actions = []
            for i in range(len(players)):
                players[i].think(obses[i])
                actions.append(network_simulator(obses[i].info["unit_valid_actions"]))
            obses = env.step(actions)
            # print(obses)
        winner = env.get_winner()
        print(winner)

    print(env.setup_commands)


if __name__ == '__main__':
   main()
