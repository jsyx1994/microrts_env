#

# system utils
import socket
from dataclasses import asdict
from typing import List, Any
from rts_wrapper.datatypes import *
import json
import numpy as np
import time
import torch

rd = np.random
rd.seed()


def action_sampler_v1(model, state, info, mode='stochastic'):
    assert mode in ['stochastic', 'deterministic']
    time_stamp = info["time_stamp"]
    # if time_stamp % 1 != 0:
    #     return []
    unit_valid_actions = info["unit_valid_actions"]  # unit and its valid actions
    height, width = info["map_size"]
    player_resources = info["player_resources"]  # global resource situation, default I'm player 0
    current_player = info["current_player"]

    spatial_feature = torch.from_numpy(state).float().unsqueeze(0)
    samples = []
    for uva in unit_valid_actions:
        u  = uva.unit
        unit_feature = torch.from_numpy(unit_feature_encoder(u, height, width)).float().unsqueeze(0)
        encoded_utt = torch.from_numpy(encoded_utt_dict[u.type]).float().unsqueeze(0)

        unit_feature = torch.cat([unit_feature, encoded_utt], dim=1)
        if mode == 'stochastic':
            sampled_unit_action = model.stochastic_action_sampler(u.type, spatial_feature, unit_feature)
        elif mode == 'deterministic':
            sampled_unit_action = model.deterministic_action_sampler(u.type, spatial_feature, unit_feature)

        samples.append((uva, sampled_unit_action))

    return network_action_translator(samples)

def get_available_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


def normalize(value, max_v, min_v):
    return 1 if max_v == min_v else (value - min_v) / (max_v - min_v)


def get_action_index(enum_action):
    return list(enum_action.__class__).index(enum_action)


def pa_to_jsonable(pas: List[PlayerAction]) -> str:
    ans = []
    for pa in pas:
        ans.append(asdict(pa))
    # json.dumps(ans)
    return json.dumps(ans)


def extract_record(gs: GameState, sl_target: int) -> Record:
    """
    get the target players' (state - actions) pair
    :param gs:
    :param sl_target:
    :return:Record
    """
    # st = time.time()
    assert sl_target in [0, 1], "target player must be No.0 or No.1!"
    units = gs.pgs.units
    actions = gs.actions
    id_unit_map = {}
    for u in units:
        id_unit_map[u.ID] = u

    t_unit = list(filter(lambda x: x.player == sl_target, units))
    t_unit_id = [u.ID for u in t_unit]
    
    # print(t_unit)
    t_uaa = [(id_unit_map[a.ID], a.action) for a in actions if a.ID in t_unit_id]
    # print(t_uaa)
    uaas = []
    for x in t_uaa:
        y = UAA(unit=x[0], unitAction=x[1])
        uaas.append(y)

    rcd = Record(
        gs=gs,
        player=sl_target,
        actions=uaas
    )
    # print(time.time() - st)
    return rcd


def state_encoder(gs: GameState, player):
    current_player = player
    # AGENT_COLLECTION
    pgs = gs.pgs
    w = pgs.width
    h = pgs.height
    units = pgs.units
    p1_info, p2_info = gs.pgs.players
    my_resources  = p1_info.resources if current_player == p1_info.ID else p2_info.resources
    opp_resources = p2_info.resources if current_player == p1_info.ID else p1_info.resources

    channel_terrain = np.array([int(x) for x in pgs.terrain]).reshape((1, h, w))
    channel_type = np.zeros((len(UNIT_COLLECTION), h, w))
    channel_hp_ratio = np.zeros((1, h, w))
    channel_resource = np.zeros((8, h, w))

    channel_is_ally = np.zeros((2, h, w))

    channel_my_resources = np.full((1, h, w), fill_value=my_resources)
    channel_opp_resources = np.full((1, h, w), fill_value=opp_resources)

    for unit in units:
        _owner = unit.player
        _type = unit.type
        _x, _y = unit.x, unit.y
        _resource_carried = unit.resources
        _hp = unit.hitpoints
        _id = unit.ID
        _one_hot_resource_pos = list(resource_encoder(_resource_carried)).index(1)
        _one_hot_type_pos = UNIT_COLLECTION.index(_type)
        _one_hot_is_ally_pos = int((current_player == _owner))
        # channel_is_ally[_player][_x][_y] = 1
        channel_hp_ratio[0][_x][_y] = _hp / UTT_DICT[_type].hp
        channel_is_ally[_one_hot_is_ally_pos][_x][_y] = 1
        channel_type[_one_hot_type_pos][_x][_y] = 1
        channel_resource[_one_hot_resource_pos][_x][_y] = 1

    spatial_features = np.vstack(
        (
            channel_is_ally,
            channel_type,
            channel_resource,
            channel_hp_ratio,
            channel_terrain,

            channel_my_resources,
            channel_opp_resources
        ),
    )
    return spatial_features


def utt_encoder(utt_str: str):
    max_value = 10000000
    min_value = -max_value

    utt = from_dict(data_class=UnitTypeTable, data=json.loads(utt_str))
    name_id_dict = {}

    encoder_dict = {}
    min_cost, max_cost = max_value, min_value
    min_hp, max_hp = max_value, min_value
    min_mindamage, max_mindamage = max_value, min_value
    min_maxdamage, max_maxdamage = max_value, min_value
    min_attackrange, max_attackrange = max_value, min_value
    min_producetime, max_producetime = max_value, min_value
    min_movetime, max_movetime = max_value, min_value
    min_attacktime, max_attacktime = max_value, min_value
    min_harvesttime, max_harvesttime = max_value, min_value
    min_returntime, max_returntime = max_value, min_value
    min_harvestamount, max_harvestamount = max_value, min_value
    min_sightradius, max_sightradius = max_value, min_value

    for ut in utt.unitTypes:
        name_id_dict[ut.name] = ut.ID

        min_cost, max_cost = min(ut.cost, min_cost), max(ut.cost, max_cost)
        min_hp, max_hp = min(ut.hp, min_hp), max(ut.hp, max_hp)
        min_mindamage, max_mindamage = min(ut.minDamage, min_mindamage), max(ut.minDamage, max_mindamage)
        min_maxdamage, max_maxdamage = min(ut.maxDamage, min_maxdamage), max(ut.maxDamage, max_maxdamage)
        min_attackrange, max_attackrange = min(ut.attackRange, min_attackrange), max(ut.attackRange,
                                                                                     max_attackrange)
        min_producetime, max_producetime = min(ut.produceTime, min_producetime), max(ut.produceTime,
                                                                                     max_producetime)
        min_movetime, max_movetime = min(ut.moveTime, min_movetime), max(ut.moveTime, max_movetime)
        min_attacktime, max_attacktime = min(ut.attackTime, min_attacktime), max(ut.attackTime, max_attacktime)
        min_harvesttime, max_harvesttime = min(ut.harvestTime, min_harvesttime), max(ut.harvestTime,
                                                                                     max_harvesttime)
        min_returntime, max_returntime = min(ut.returnTime, min_returntime), max(ut.returnTime, max_returntime)
        min_harvestamount, max_harvestamount = min(ut.harvestAmount, min_harvestamount), max(ut.harvestAmount,
                                                                                             max_harvestamount)
        min_sightradius, max_sightradius = min(ut.sightRadius, min_sightradius), max(ut.sightRadius,
                                                                                     max_sightradius)

    for ut in utt.unitTypes:
        # id
        id = np.zeros(7)
        id[ut.ID] = 1
        cost = np.array([normalize(ut.cost, max_cost, min_cost)])
        hp = np.array([normalize(ut.hp, max_hp, min_hp)])
        min_damage = np.array([normalize(ut.minDamage, max_mindamage, min_mindamage)])
        max_damage = np.array([normalize(ut.maxDamage, max_maxdamage, min_maxdamage)])
        attack_range = np.array([normalize(ut.attackRange, max_attackrange, min_attackrange)])
        produce_time = np.array([normalize(ut.produceTime, max_producetime, min_producetime)])
        move_time = np.array([normalize(ut.moveTime, max_movetime, min_movetime)])
        attack_time = np.array([normalize(ut.attackTime, max_attacktime, min_attacktime)])
        harvest_time = np.array([normalize(ut.harvestTime, max_harvesttime, min_harvesttime)])
        return_time = np.array([normalize(ut.returnTime, max_returntime, min_returntime)])
        harvest_amount = np.array([normalize(ut.harvestAmount, max_harvestamount, min_harvestamount)])
        sight_radius = np.array([normalize(ut.sightRadius, max_sightradius, min_sightradius)])
        # produces
        produces = np.zeros(7)
        for name in ut.produces:
            produces[name_id_dict[name]] = 1
        # produced_by
        produced_by = np.zeros(7)
        for name in ut.producedBy:
            produced_by[name_id_dict[name]] = 1

        is_resource = np.zeros(2)
        if not ut.isResource:
            is_resource[0] = 1
        else:
            is_resource[1] = 1

        is_stockplie = np.zeros(2)
        if not ut.isStockpile:
            is_stockplie[0] = 1
        else:
            is_stockplie[1] = 1

        can_harvest = np.zeros(2)
        if not ut.canHarvest:
            can_harvest[0] = 1
        else:
            can_harvest[1] = 1

        can_move = np.zeros(2)
        if not ut.canMove:
            can_move[0] = 1
        else:
            can_move[1] = 1

        can_attack = np.zeros(2)
        if not ut.canAttack:
            can_attack[0] = 1
        else:
            can_attack[1] = 1

        ans = np.hstack(
            (id,  # 7
             cost,  # 1
             hp,  # 2
             min_damage,  # 1
             max_damage,  # 1
             attack_range,  # 1
             produce_time,  # 1
             move_time,  # 1
             attack_time,  # 1
             harvest_time,  # 1
             return_time,  # 1
             harvest_amount,  # 1
             sight_radius,  # 1
             is_resource,  # 2
             is_stockplie,  # 2
             can_harvest,  # 2
             can_move,  # 2
             can_attack,  # 2
             produces,  # 7
             produced_by  # 7
             )
        )
        encoder_dict[ut.name] = ans

    return encoder_dict, ans.size


# def unit_instance_encoder(map_height, map_width, unit: Unit):
#     owner = np.zeros(2)
#     owner[unit.player] = 1
#     x_pos = np.array([unit.x / map_width])
#     y_pos = np.array([unit.y / map_height])
#     hp_ratio = np.array([unit.hitpoints / UTT_DICT[unit.type].hp])
#     resource = resource_encoder(unit.resources)
#     feature = np.hstack((
#         owner,
#         x_pos,
#         y_pos,
#         resource,
#         hp_ratio,
#     ))
#     return feature


def network_action_translator(unit_validaction_choices) -> List[PlayerAction]:
    """
    translate network actions to ones game readable
    :param unit_validaction_choices: tuple of ([unit with valid action list], [unitAction instance form datatypes])
    :return:
    """
    pas = []
    for uva, choice in unit_validaction_choices:
        unit = uva.unit
        valid_actions = uva.unitActions
        pa = PlayerAction(unitID=unit.ID)

        def get_valid_probe_action(direction) -> UnitAction:
            """
            probe doesn't include produce action
            :param direction:
            :return:
            """
            # probe actions includes: move(parameter), attack(x, y),
            ua = [action for action in valid_actions if
                  (action.parameter == direction.value and action.type != ACTION_TYPE_PRODUCE) or
                  (action.x == unit.x + DIRECTION_OFFSET_X[direction.value] and
                   action.y == unit.y + DIRECTION_OFFSET_Y[direction.value])
                  ]

            assert len(ua) <= 1
            return ua[0] if ua else UnitAction()

        def get_valid_produce_directions(unit_type) -> List:
            """
            if the object to be produce have direction parameter in valid_actions
            :param unit_type:
            :return:
            """
            return [action.parameter for action in valid_actions if action.unitType == unit_type]

        def issue_produce(unit_type_name):
            """
            checking if the unit to produce is valid
            :param unit_type_name: the unit name of the object to be produced
            :return:
            """
            directions = get_valid_produce_directions(unit_type_name)
            if not directions:
                # print("Invalid network action, do nothing")
                pa.unitAction.type = ACTION_TYPE_NONE
            else:
                pa.unitAction.type = ACTION_TYPE_PRODUCE
                pa.unitAction.unitType = unit_type_name
                pa.unitAction.parameter = int(rd.choice(directions))

        if unit.type == UNIT_TYPE_NAME_BASE:
            if choice == BaseAction.DO_NONE:
                pa.unitAction.type = ACTION_TYPE_NONE

            elif choice == BaseAction.DO_LAY_WORKER:
                issue_produce(UNIT_TYPE_NAME_WORKER)

        elif unit.type == UNIT_TYPE_NAME_BARRACKS:
            if choice == BarracksAction.DO_NONE:
                pa.unitAction.type = ACTION_TYPE_NONE

            elif choice == BarracksAction.DO_LAY_LIGHT:
                issue_produce(UNIT_TYPE_NAME_LIGHT)

            elif choice == BarracksAction.DO_LAY_HEAVY:
                issue_produce(UNIT_TYPE_NAME_HEAVY)

            elif choice == BarracksAction.DO_LAY_RANGED:
                issue_produce(UNIT_TYPE_NAME_RANGED)

        elif unit.type == UNIT_TYPE_NAME_WORKER:
            if choice == WorkerAction.DO_NONE:
                pa.unitAction.type = ACTION_TYPE_NONE

            elif choice == WorkerAction.DO_UP_PROBE:
                pa.unitAction = get_valid_probe_action(WorkerAction.DO_UP_PROBE)

            elif choice == WorkerAction.DO_RIGHT_PROBE:
                pa.unitAction = get_valid_probe_action(WorkerAction.DO_RIGHT_PROBE)

            elif choice == WorkerAction.DO_DOWN_PROBE:
                pa.unitAction = get_valid_probe_action(WorkerAction.DO_DOWN_PROBE)

            elif choice == WorkerAction.DO_LEFT_PROBE:
                pa.unitAction = get_valid_probe_action(WorkerAction.DO_LEFT_PROBE)

            elif choice == WorkerAction.DO_LAY_BASE:
                issue_produce(UNIT_TYPE_NAME_BASE)

            elif choice == WorkerAction.DO_LAY_BARRACKS:
                issue_produce(UNIT_TYPE_NAME_BARRACKS)

        elif unit.type == UNIT_TYPE_NAME_LIGHT:
            if choice == LightAction.DO_NONE:
                pa.unitAction.type = ACTION_TYPE_NONE

            elif choice == LightAction.DO_UP_PROBE:
                pa.unitAction = get_valid_probe_action(LightAction.DO_UP_PROBE)

            elif choice == LightAction.DO_RIGHT_PROBE:
                pa.unitAction = get_valid_probe_action(LightAction.DO_RIGHT_PROBE)

            elif choice == LightAction.DO_DOWN_PROBE:
                pa.unitAction = get_valid_probe_action(LightAction.DO_DOWN_PROBE)

            elif choice == LightAction.DO_LEFT_PROBE:
                pa.unitAction = get_valid_probe_action(LightAction.DO_LEFT_PROBE)

        elif unit.type == UNIT_TYPE_NAME_HEAVY:
            if choice == HeavyAction.DO_NONE:
                pa.unitAction.type = ACTION_TYPE_NONE

            elif choice == HeavyAction.DO_UP_PROBE:
                pa.unitAction = get_valid_probe_action(HeavyAction.DO_UP_PROBE)

            elif choice == HeavyAction.DO_RIGHT_PROBE:
                pa.unitAction = get_valid_probe_action(HeavyAction.DO_RIGHT_PROBE)

            elif choice == HeavyAction.DO_DOWN_PROBE:
                pa.unitAction = get_valid_probe_action(HeavyAction.DO_DOWN_PROBE)

            elif choice == HeavyAction.DO_LEFT_PROBE:
                pa.unitAction = get_valid_probe_action(HeavyAction.DO_LEFT_PROBE)

        elif unit.type == UNIT_TYPE_NAME_RANGED:
            pass

        # if pa.unitAction.type == ACTION_TYPE_ATTACK_LOCATION:
        #     input()
        pas.append(pa)
    return pas


# TODO: test the following
def game_action_translator(u: Unit, ua: UnitAction):
    """
    translate the game actions to ones network readable
    :return: network action
    """
    def attack_trans(x, y, _x, _y):
        for i in range(4):
            if (x + DIRECTION_OFFSET_X[i] == _x) and (y + DIRECTION_OFFSET_Y[i] == _y):
                return i


    if   u.type == UNIT_TYPE_NAME_BASE:

        if   ua.type == ACTION_TYPE_NONE:
            return get_action_index(BaseAction.DO_NONE)

        elif ua.type == ACTION_TYPE_PRODUCE:
            return get_action_index(BaseAction.DO_LAY_WORKER)

    # barracks
    elif u.type == UNIT_TYPE_NAME_BARRACKS:

        if   ua.type == ACTION_TYPE_NONE:
            return get_action_index(BarracksAction.DO_NONE)

        elif ua.type == ACTION_TYPE_PRODUCE:
            if   ua.unitType == UNIT_TYPE_NAME_LIGHT:
                return get_action_index(BarracksAction.DO_LAY_LIGHT)
            elif ua.unitType == UNIT_TYPE_NAME_HEAVY:
                return get_action_index(BarracksAction.DO_LAY_HEAVY)
            elif ua.unitType == UNIT_TYPE_NAME_RANGED:
                return get_action_index(BarracksAction.DO_LAY_RANGED)

    # worker
    elif u.type == UNIT_TYPE_NAME_WORKER:

        if   ua.type == ACTION_TYPE_NONE:
            return get_action_index(WorkerAction.DO_NONE)

        elif ua.type in (ACTION_TYPE_MOVE, ACTION_TYPE_HARVEST, ACTION_TYPE_RETURN):
            return get_action_index(WorkerAction(ua.parameter))

        elif ua.type == ACTION_TYPE_ATTACK_LOCATION:
            return get_action_index(WorkerAction(attack_trans(u.x, u.y, ua.x, ua.y)))

        elif ua.type == ACTION_TYPE_PRODUCE:
            if ua.unitType == UNIT_TYPE_NAME_BASE:
                return get_action_index(WorkerAction.DO_LAY_BASE)
            elif ua.unitType == UNIT_TYPE_NAME_BARRACKS:
                return get_action_index(WorkerAction.DO_LAY_BARRACKS)

    # light
    elif u.type == UNIT_TYPE_NAME_LIGHT:
        if ua.type == ACTION_TYPE_NONE:
            return get_action_index(LightAction.DO_NONE)
        elif ua.type in (ACTION_TYPE_MOVE, ACTION_TYPE_HARVEST, ACTION_TYPE_RETURN):
            return get_action_index(LightAction(ua.parameter))
        elif ua.type == ACTION_TYPE_ATTACK_LOCATION:
            return get_action_index(LightAction(attack_trans(u.x, u.y, ua.x, ua.y)))

    # heavy
    elif u.type == UNIT_TYPE_NAME_HEAVY:
        if ua.type == ACTION_TYPE_NONE:
            return get_action_index(HeavyAction.DO_NONE)
        elif ua.type in (ACTION_TYPE_MOVE, ACTION_TYPE_HARVEST, ACTION_TYPE_RETURN):
            return get_action_index(HeavyAction(ua.parameter))
        elif ua.type == ACTION_TYPE_ATTACK_LOCATION:
            return get_action_index(HeavyAction(attack_trans(u.x, u.y, ua.x, ua.y)))

    # ranged
    elif u.type == UNIT_TYPE_NAME_RANGED:
        pass

    return ACTION_TYPE_NONE


def resource_encoder(amount, feature_length=8, amount_threshold=2):
    resource = np.zeros(8)
    if amount == 0:
        resource[0] = 1
        return resource
    bit_pos = 1
    # amount_threshold = 2
    for _ in range(1, feature_length):
        if amount > amount_threshold:
            bit_pos += 1
            amount_threshold *= 2
        else:
            resource[bit_pos] = 1
            return resource

    resource[-1] = 1
    return resource


def unit_feature_encoder(unit:Unit, map_height, map_width):
    unit_type = unit.type
    unit_x = unit.x
    unit_y = unit.y
    unit_resource = unit.resources
    unit_hp = unit.hitpoints

    type_feature = np.zeros(len(UNIT_COLLECTION))
    type_feature[UNIT_COLLECTION.index(unit_type)] = 1

    x_ratio_feature = np.array([unit_x / map_width])
    y_ratio_feature = np.array([unit_y / map_height])
    resource_feature = resource_encoder(unit_resource)
    hp_ratio_feature = np.array([unit_hp / UTT_DICT[unit_type].hp])

    unit_feature = np.hstack(
        (
            type_feature,
            x_ratio_feature,
            y_ratio_feature,
            resource_feature,
            hp_ratio_feature
        )
    )
    return unit_feature


# def demo_unit_instance_encoder():
#     unit_str = '{"type":"Base", "ID":20, "player":0, "x":2, "y":2, "resources":0, "hitpoints":10}'
#     unit = from_dict(data_class=Unit, data=json.loads(unit_str))
#     print(unit_instance_encoder(8, 8, unit))


def test_state_encoder():
    str_gs = '{"reward":150.0,"done":false,"validActions":[{"unit":{"type":"Worker", "ID":22, "player":0, "x":0, "y":4, "resources":0, "hitpoints":1},"unitActions":[{"type":1, "parameter":0} ,{"type":1, "parameter":2} ,{"type":0, "parameter":10}]},{"unit":{"type":"Worker", "ID":24, "player":0, "x":3, "y":4, "resources":0, "hitpoints":1},"unitActions":[{"type":1, "parameter":0} ,{"type":1, "parameter":1} ,{"type":1, "parameter":2} ,{"type":1, "parameter":3} ,{"type":0, "parameter":10}]}],"gs":{"time":187,"pgs":{"width":6,"height":6,"terrain":"000000000000000000000000000000000000","players":[{"ID":0, "resources":2},{"ID":1, "resources":5}],"units":[{"type":"Resource", "ID":0, "player":-1, "x":0, "y":0, "resources":229, "hitpoints":1},{"type":"Base", "ID":19, "player":1, "x":5, "y":5, "resources":0, "hitpoints":10},{"type":"Base", "ID":20, "player":0, "x":2, "y":2, "resources":0, "hitpoints":10},{"type":"Worker", "ID":22, "player":0, "x":0, "y":4, "resources":0, "hitpoints":1},{"type":"Worker", "ID":23, "player":0, "x":5, "y":4, "resources":0, "hitpoints":1},{"type":"Worker", "ID":24, "player":0, "x":3, "y":4, "resources":0, "hitpoints":1},{"type":"Worker", "ID":25, "player":0, "x":0, "y":1, "resources":1, "hitpoints":1},{"type":"Worker", "ID":26, "player":0, "x":1, "y":4, "resources":0, "hitpoints":1}]},"actions":[{"ID":20, "time":153, "action":{"type":4, "parameter":1, "unitType":"Worker"}},{"ID":26, "time":178, "action":{"type":1, "parameter":1}},{"ID":19, "time":180, "action":{"type":0, "parameter":10}},{"ID":25, "time":182, "action":{"type":1, "parameter":1}},{"ID":23, "time":185, "action":{"type":1, "parameter":3}}]}}'
    gs_wrapper = from_dict(data_class=GsWrapper, data=json.loads(str_gs))
    print(extract_record(gs_wrapper.gs, 0))


def test_resource_encoder():
    print(resource_encoder(0))  # [1. 0. 0. 0. 0. 0. 0. 0.]
    print(resource_encoder(1))  # [0. 1. 0. 0. 0. 0. 0. 0.]
    print(resource_encoder(2))  # [0. 1. 0. 0. 0. 0. 0. 0.]
    print(resource_encoder(4))  # [0. 1. 0. 0. 0. 0. 0. 0.]
    print(resource_encoder(17))  # [0. 1. 0. 0. 0. 0. 0. 0.]
    print(resource_encoder(63))  # [0. 0. 0. 0. 0. 0. 1. 0.]
    print(resource_encoder(14453))  # [0. 0. 0. 0. 0. 0. 0. 1.]


encoded_utt_dict, encoded_utt_feature_size = utt_encoder(UTT_ORI)

if __name__ == '__main__':
   test_state_encoder()
    # test_resource_encoder()