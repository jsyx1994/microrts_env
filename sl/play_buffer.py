import numpy as np
from rts_wrapper.envs.utils import state_encoder, unit_feature_encoder, game_action_translator
from rts_wrapper.datatypes import *
from dataclasses import dataclass
from numpy import random

@dataclass
class Sample:
    gs      : GameState
    t_player: int
    unit    : Unit
    action  : UnitAction


class PlayBuffer(object):
    def __init__(self, size=100000):
        # self._storage = {
        #     UNIT_TYPE_NAME_BASE:        [],
        #     UNIT_TYPE_NAME_BARRACKS:    [],
        #     UNIT_TYPE_NAME_WORKER:      [],
        #     UNIT_TYPE_NAME_LIGHT:       [],
        #     UNIT_TYPE_NAME_HEAVY:       [],
        #     UNIT_TYPE_NAME_RANGED:      [],
        # }
        self._storage = []
        self._maxsize = size

    def __len__(self):
        return len(self._storage)

    def _encode_sample(self, idxes):
        # print(idxes)
        states, unit_types, units, actions = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            state, t_player, unit, action = data.gs, data.t_player, data.unit, data.action
            # if action.type == 5:
            #     input()
            states.append(state_encoder(gs=state, player=t_player))
            units.append(unit_feature_encoder(unit, state.pgs.height, state.pgs.width))
            actions.append(game_action_translator(unit, action))
            unit_types.append(unit.type)

        return np.array(states), np.array(unit_types), np.array(units), np.array(actions)

    def _factorize(self, batch_size, batch) -> dict:
        ans = {
                UNIT_TYPE_NAME_BASE:        [],
                UNIT_TYPE_NAME_BARRACKS:    [],
                UNIT_TYPE_NAME_WORKER:      [],
                UNIT_TYPE_NAME_LIGHT:       [],
                UNIT_TYPE_NAME_HEAVY:       [],
                UNIT_TYPE_NAME_RANGED:      [],
        }
        states, unit_types, units, actions = batch
        for i in range(batch_size):
            ans[unit_types[i]].append((states[i], units[i], actions[i]))

        for key in ans:
            states, units, actions = [], [], []
            for v in ans[key]:
                states.append(v[0])
                units.append(v[1])
                actions.append(v[2])
            ans[key] = np.array(states), np.array(units), np.array(actions)

        return ans

    def push(self, gs: GameState, t_player: int, unit: Unit, action: UnitAction):
        data = Sample(gs=gs, t_player=t_player, unit=unit, action=action)
        self._storage.append(data)

    @property
    def _bases(self):
        return filter(lambda x: x.unit.type == UNIT_TYPE_NAME_BASE, self._storage)

    @property
    def _workers(self):
        return filter(lambda x: x.unit.type == UNIT_TYPE_NAME_WORKER, self._storage)

    @property
    def _more_properties(self):
        return None

    def sample(self, batch_size, factorize=True):
        idxes = [random.randint(0, len(self._storage)) for _ in range(batch_size)]
        encoded_samples = self._encode_sample(idxes)
        return self._factorize(batch_size, encoded_samples) if factorize else encoded_samples


if __name__ == '__main__':
    pass