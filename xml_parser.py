import xml.etree.ElementTree as ET
from dacite import from_dict
import json
import xmltodict
from rts_wrapper.envs.utils import fix_keys
from rts_wrapper.datatypes import *


def parse_xml():
    tree = ET.parse("/home/toby/game.xml")
    xdict = xmltodict.parse(ET.tostring(tree.getroot()))
    # xdict = xmltodict.parse(raw_str)
    return xdict['rts.Trace']['entries']['rts.TraceEntry']


def parse_gs(odict):
    gs  = fix_keys(odict)

    gs['pgs'] = gs.pop('rts.PhysicalGameState')
    pgs = fix_keys(gs['pgs'])

    players = pgs['players'] = pgs['players'].pop('rts.Player')
    for p in players:
        fix_keys(p)

    units = pgs['units'] = pgs['units'].pop('rts.units.Unit')
    for u in units:
        fix_keys(u)


    actions = gs['actions']['action']
    for a in actions:
        fix_keys(a)
        a['action'] = fix_keys(a.pop('UnitAction'))

    input()
    # gs = json.loads(json.dumps(gs))
    print(gs)
    from_dict(data_class=GameState, data=gs)
    # print(odict)


def main():
    for x in parse_xml()[4:]:
        parse_gs(x)


if __name__ == '__main__':
    main()