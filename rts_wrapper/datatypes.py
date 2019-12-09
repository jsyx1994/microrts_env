from dataclasses import dataclass, asdict
from typing import List, Any, Dict, Optional
from enum import Enum
from dacite import from_dict
import json

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
ACTION_PARAMETER_VALID_DIRECTION_NUMS = 4

UNIT_TYPE_NAME_BASE = 'Base'
UNIT_TYPE_NAME_BARRACKS = 'Barracks'
UNIT_TYPE_NAME_WORKER = 'Worker'
UNIT_TYPE_NAME_LIGHT = 'Light'
UNIT_TYPE_NAME_HEAVY = 'Heavy'
UNIT_TYPE_NAME_RANGED = 'Ranged'
UNIT_TYPE_NAME_RESOURCE = 'Resource'

AGENT_COLLECTION = [UNIT_TYPE_NAME_BASE, UNIT_TYPE_NAME_BARRACKS, UNIT_TYPE_NAME_WORKER, UNIT_TYPE_NAME_LIGHT,
                    UNIT_TYPE_NAME_HEAVY, UNIT_TYPE_NAME_RANGED]


# unit collection is used in the real state encoding
UNIT_COLLECTION = AGENT_COLLECTION + ['Resource']

DIRECTION_OFFSET_X = [0, 1, 0, -1]
DIRECTION_OFFSET_Y = [-1, 0, 1, 0]

UTT_ORI = '{"moveConflictResolutionStrategy":3,"unitTypes":[{"ID":0, "name":"Resource", "cost":1, "hp":1, ' \
          '"minDamage":1, "maxDamage":1, "attackRange":1, "produceTime":10, "moveTime":10, "attackTime":10, ' \
          '"harvestTime":10, "returnTime":10, "harvestAmount":1, "sightRadius":0, "isResource":true, ' \
          '"isStockpile":false, "canHarvest":false, "canMove":false, "canAttack":false, "produces":[], "producedBy":[' \
          ']}, {"ID":1, "name":"Base", "cost":10, "hp":10, "minDamage":1, "maxDamage":1, "attackRange":1, ' \
          '"produceTime":250, "moveTime":10, "attackTime":10, "harvestTime":10, "returnTime":10, "harvestAmount":1, ' \
          '"sightRadius":5, "isResource":false, "isStockpile":true, "canHarvest":false, "canMove":false, ' \
          '"canAttack":false, "produces":["Worker"], "producedBy":["Worker"]}, {"ID":2, "name":"Barracks", "cost":5, ' \
          '"hp":4, "minDamage":1, "maxDamage":1, "attackRange":1, "produceTime":200, "moveTime":10, "attackTime":10, ' \
          '"harvestTime":10, "returnTime":10, "harvestAmount":1, "sightRadius":3, "isResource":false, ' \
          '"isStockpile":false, "canHarvest":false, "canMove":false, "canAttack":false, "produces":["Light", "Heavy", ' \
          '"Ranged"], "producedBy":["Worker"]}, {"ID":3, "name":"Worker", "cost":1, "hp":1, "minDamage":1, ' \
          '"maxDamage":1, "attackRange":1, "produceTime":50, "moveTime":10, "attackTime":5, "harvestTime":20, ' \
          '"returnTime":10, "harvestAmount":1, "sightRadius":3, "isResource":false, "isStockpile":false, ' \
          '"canHarvest":true, "canMove":true, "canAttack":true, "produces":["Base", "Barracks"], "producedBy":[' \
          '"Base"]}, {"ID":4, "name":"Light", "cost":2, "hp":4, "minDamage":2, "maxDamage":2, "attackRange":1, ' \
          '"produceTime":80, "moveTime":8, "attackTime":5, "harvestTime":10, "returnTime":10, "harvestAmount":1, ' \
          '"sightRadius":2, "isResource":false, "isStockpile":false, "canHarvest":false, "canMove":true, ' \
          '"canAttack":true, "produces":[], "producedBy":["Barracks"]}, {"ID":5, "name":"Heavy", "cost":2, "hp":4, ' \
          '"minDamage":4, "maxDamage":4, "attackRange":1, "produceTime":120, "moveTime":12, "attackTime":5, ' \
          '"harvestTime":10, "returnTime":10, "harvestAmount":1, "sightRadius":2, "isResource":false, ' \
          '"isStockpile":false, "canHarvest":false, "canMove":true, "canAttack":true, "produces":[], "producedBy":[' \
          '"Barracks"]}, {"ID":6, "name":"Ranged", "cost":2, "hp":1, "minDamage":1, "maxDamage":1, "attackRange":3, ' \
          '"produceTime":100, "moveTime":10, "attackTime":5, "harvestTime":10, "returnTime":10, "harvestAmount":1, ' \
          '"sightRadius":3, "isResource":false, "isStockpile":false, "canHarvest":false, "canMove":true, ' \
          '"canAttack":true, "produces":[], "producedBy":["Barracks"]}]} '


@dataclass
class Observations:
    reward: float
    observation: Any
    done: bool
    info: Dict


class BaseAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_BASE

    DO_NONE = 0
    DO_LAY_WORKER = 1


class BarracksAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_BARRACKS

    DO_NONE = 0
    DO_LAY_LIGHT = 1
    DO_LAY_HEAVY = 2
    DO_LAY_RANGED = 3

    @staticmethod
    def get_index(action):
        return list(BarracksAction).index(action)


class WorkerAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_WORKER

    DO_NONE = -1

    # 0, 1, 2, 3 must agree with up, right, down, left
    # need to convert to valid actions using unit_valid_action according to the specific condition.
    # example:DO_UP_PROBE: walk up when no obstacles, but attack when enemy
    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3

    # produce: randomly pick directions
    DO_LAY_BASE = 4  # type4 unitType:base
    DO_LAY_BARRACKS = 5  # type4 unitType:barracks


class LightAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_LIGHT
    DO_NONE = -1

    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3


class HeavyAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_HEAVY
    DO_NONE = -1

    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3


class RangedAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_RANGED

    DO_NONE = -1

    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3

    DO_ATTACK_NEAREST = 4  # need java coding
    DO_ATTACK_WEAKEST = 5


action_collection = [BaseAction, BarracksAction, WorkerAction, LightAction, HeavyAction, RangedAction]

AGENT_ACTIONS_MAP = {}
for _action in action_collection:
    AGENT_ACTIONS_MAP[_action.__type_name__] = _action


@dataclass
class UnitType:
    ID: int
    name: str
    cost: int
    hp: int
    minDamage: int
    maxDamage: int
    attackRange: int
    produceTime: int
    moveTime: int
    attackTime: int
    harvestTime: int
    returnTime: int
    harvestAmount: int
    sightRadius: int
    isResource: bool
    isStockpile: bool
    canHarvest: bool
    canMove: bool
    canAttack: bool
    produces: List[str]
    producedBy: List[str]


@dataclass
class UnitTypeTable:
    moveConflictResolutionStrategy: int
    unitTypes: List[UnitType]


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
class UnitAction:
    type: Optional[int] = ACTION_TYPE_NONE
    parameter: Optional[int] = ACTION_PARAMETER_DIRECTION_NONE
    unitType: Optional[str] = ""
    x: Optional[int] = -1
    y: Optional[int] = -1


@dataclass
class AssignedAction:
    ID: int
    action: UnitAction
    time: Optional[int] = 0


@dataclass
class GameState:
    time: int
    pgs: Pgs
    actions: List[AssignedAction]


@dataclass
class Config:
    ai1_type: str
    ai2_type: str
    map_path: str
    height: int
    width: int
    self_play: Optional[bool] = False
    max_cycles: Optional[int] = 5000
    max_episodes: Optional[int] = 10000
    period: Optional[int] = 5
    render: Optional[int] = 1
    utt: Optional[dict] = from_dict(data_class=UnitTypeTable, data=json.loads(UTT_ORI))
    # auto_port: Optional[bool] = False
    # client_port: Optional[int] = 0
    microrts_path: Optional[str] = "~/microrts_env"
    microrts_repo_path: Optional[str] = ""
    client_ip: Optional[str] = "127.0.0.1"

    window_size: Optional[int] = 1
    evaluation_filename: Optional[str] = ""
    frame_skip: Optional[int] = 0



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
    unitAction: Optional[UnitAction] = UnitAction()


# for sl
@dataclass
class UAA:
    unit: Unit
    unitAction: UnitAction


@dataclass
class Record:
    """
    all actions under one game state given the supervised target player
    """
    gs: GameState
    player: int
    actions: List[UAA]


@dataclass
class Records:
    records: List[Record]
# /for sl


UTT_DICT = {}
for t in from_dict(data_class=UnitTypeTable, data=json.loads(UTT_ORI)).unitTypes:
    UTT_DICT[t.name] = t

# print(UTT_DICT)

if __name__ == '__main__':
    print(list(LightAction.DO_NONE.__class__))
