from dataclasses import dataclass, asdict
from typing import List, Any, Dict, Optional
from enum import Enum, auto

ACTION_TYPE_NONE = 0
ACTION_TYPE_MOVE = 1
ACTION_TYPE_HARVEST = 2
ACTION_TYPE_RETURN = 3
ACTION_TYPE_PRODUCE = 4
ACTION_TYPE_ATTACK_LOCATION = 5

ACTION_PARAMETER_DIRECTION_NONE = -1
ACTION_PARAMETER_DIRECTION_UP = 0
ACTION_PARAMETER_DIRECTION_RIGHT = 1
ACTION_PARAMETER_DIRECTION_DOWN = 2
ACTION_PARAMETER_DIRECTION_LEFT = 3

# @dataclass
# class LearningSignal:
#     reward: float
#     observation: List[List[List[int]]]
#     done: bool
#     info: Dict


class BaseAction(Enum):
    DO_NONE = 0
    DO_PRODUCE = 1

    ACTION_NUMBER = 2


class WorkerActon(Enum):
    DO_NONE = -1

    # need to convert to valid actions using unit_valid_action according to the specific condition.
    # example:DO_UP_PROBE: walk up when no obstacles, but attack when enemy
    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3

    # produce: randomly pick directions
    DO_LAY_BASE = 4         # type4 unitType:base
    DO_LAY_BARRACKS = 5     # type4 unitType:barracks


class MeleeAction(Enum):
    DO_NONE = -1

    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3


class ArcherAction(Enum):
    DO_NONE = -1

    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3

    DO_ATTACK_NEAREST = 4   # need java coding
    DO_ATTACK_WEAKEST = 5





@dataclass
class UnitType:
    id: int
    name: str
    cost: int
    hp: int
    min_damage: int
    max_damage: int
    attack_range: int
    produce_time: int
    move_time: int
    attack_time: int
    harvest_time: int
    return_time: int
    harvest_amount: int
    sight_radius: int
    is_resource: bool
    is_stockpile: bool
    can_harvest: bool
    can_move: bool
    can_attack: bool
    produces: List[str]
    produced_by: List[str]


@dataclass
class GameInfo:
    move_conflict_resolution_strategy: int
    unit_types: List[UnitType]


@dataclass
class Player:
    ID: int
    resources: int


@dataclass
class Unit:
    type: str
    ID: int
    player: int
    x: int
    y: int
    resources: int
    hitpoints: int


@dataclass
class Pgs:
    width: int
    height: int
    terrain: str
    players: List[Player]
    units: List[Unit]


@dataclass
class GameState:
    time: int
    pgs: Pgs
    actions: List[Any]

@dataclass
class Config:
    ai1_type: str
    ai2_type: str
    map_path: str
    max_cycles: Optional[int] = 5000
    period: Optional[int] = 5
    render: Optional[bool] = True
    # auto_port: Optional[bool] = False
    client_port: Optional[int] = 0
    microrts_path: Optional[str] = ""
    microrts_repo_path: Optional[str] = ""
    maximum_t: Optional[int] = 2000
    client_ip: Optional[str] = "127.0.0.1"
    height: Optional[int] = 0
    width: Optional[int] = 0
    window_size: Optional[int] = 1
    evaluation_filename: Optional[str] = ""
    frame_skip: Optional[int] = 0

@dataclass
class UnitAction:
    type: Optional[int] = ACTION_TYPE_NONE
    parameter: Optional[int] = ACTION_PARAMETER_DIRECTION_NONE
    unitType: Optional[str] = ""
    x: Optional[int] = -1
    y: Optional[int] = -1

@dataclass
class UnitValidAction:
    unit: Unit
    unitActions: List[UnitAction]

@dataclass
class GsWrapper:
    gs: GameState
    reward: float
    validActions: List[UnitValidAction]
    done: Optional[bool] = False


@dataclass
class PlayerAction:
    unitID: int
    unitAction: UnitAction


if __name__ == '__main__':
    for name, member in ArcherAction.__members__.items():
        print(name, member)
