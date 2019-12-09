from rts_wrapper import Config
from rts_wrapper.envs import MicroRts
import gym
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from rts_wrapper.envs import BaseEnv
from rts_wrapper.envs.player import Player
from rts_wrapper.envs.utils import get_available_port, network_simulator
from rts_wrapper.datatypes import Observations
from dacite import from_dict


class BattleEnv(BaseEnv):
    players = []
    _p_threads_pool = None

    @property
    def players_num(self):
        return len(self.players)

    def __init__(self, config):
        super(BattleEnv, self).__init__(config)
        self._counting_players()
        self._p_threads_pool = Pool(len(self.players))

        # do anything done before calling start_client
        Process(target=self.start_client).start()
        self._players_join()

    def _counting_players(self):
        if self.ai1_type.startswith("socket"):
            player = Player(0, self.config.client_ip, get_available_port())
            self._add_commands("--port" + str(1 + player.id), str(player.port))
            self.players.append(player)
        if self.ai2_type.startswith("socket"):
            player = Player(1, self.config.client_ip, get_available_port())
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
        st = time.time()
        print(time.time() - st)

        thread_num = len(self.players)
        signals_threads = [self._p_threads_pool.apply_async(self.players[i].act, (action[i],)) for i in range(thread_num)]
        signals_res = [res.get() for res in signals_threads]

        st = time.time()
        # self._p_threads_pool.terminate()

        print(time.time() - st)

        # signals_res = []
        # with ProcessPoolExecutor(max_workers=thread_num) as e:
        #     for i in range(thread_num):
        #         future = e.submit(self.players[i].act, action[i])
        #         signals_res.append(future.result())
        # # print(signals_res)
        return [self._obs2dataclass(sig) for sig in signals_res]

    def reset(self, **kwargs):
        signals = []
        for player in self.players:
            signals.append(self._obs2dataclass(player.reset()))
        return signals

    def get_winner(self):
        """
        this function must come after the done is true (i.e. game is over)
        -1: tie
        0: player 0
        1: player 1
        :return:
        """
        return self.conn.recv(1024).decode('utf-8')


def main():
    env = gym.make("SelfPlayOneWorkerAndBaseWithResources-v0")
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
        for i in range(len(players)):
            winner = env.get_winner()

    print(env.setup_commands)

if __name__ == '__main__':
   main()
