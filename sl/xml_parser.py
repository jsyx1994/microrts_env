import xml.etree.ElementTree as ET
from dacite import from_dict
import json
import xmltodict
from rts_wrapper.datatypes import GameState, Records
import os
from os import listdir
from zipfile import  ZipFile
from rts_wrapper.envs.utils import extract_record
import time
import dill
base_dir = '~/'
tournament_dir = os.path.join(base_dir, "tournament_5/traces")
saving_dir = os.path.join(base_dir, "rcds.pck")


def fix_keys(odict):
    """
    only used to fix keys form xml files
    :param odict:
    :return:
    """
    keys = [k for k in odict]
    for key in keys:
        odict[key.split('@')[-1]] = odict.pop(key)
    # print(dic)
    return odict


def parse_xml(xml_bstr):
    # tree = ET.parse(path)
    # xdict = xmltodict.parse(ET.tostring(tree.getroot()))
    st = time.time()
    xdict = xmltodict.parse(xml_bstr)
    # print(time.time() - st)
    return xdict['rts.Trace']['entries']['rts.TraceEntry']


def parse_gs(gs):
    """
    parse game state from traces
    :param gs:
    :return:
    """
    st = time.time()
    gs = json.loads(json.dumps(gs))

    gs = fix_keys(gs)
    gs['pgs'] = gs.pop('rts.PhysicalGameState')
    pgs = fix_keys(gs['pgs'])

    players = pgs['players'] = pgs['players'].pop('rts.Player')
    for p in players:
        fix_keys(p)
        p['ID'] = int(p['ID'])
        p['resources'] = int(p['resources'])

    units = pgs['units'] = pgs['units'].pop('rts.units.Unit')
    for u in units:
        fix_keys(u)
        u['ID'] = int(u['ID'])
        u['player'] = int(u['player'])
        u['x'] = int(u['x'])
        u['y'] = int(u['y'])
        u['resources'] = int(u['resources'])
        u['hitpoints'] = int(u['hitpoints'])

    if gs['actions'] is None:
        gs['actions'] = []
    else:
        gs['actions'] = gs['actions'].pop('action')
    # actions = gs['actions']['action']
    if isinstance(gs['actions'], dict):
        gs['actions'] = [gs['actions']]
    # print('actions:', gs['actions'])
    for a in gs['actions']:
        fix_keys(a)
        # print(a)s
        a['ID'] = int(a.pop('unitID'))
        a['action'] = a.pop('UnitAction')
        fix_keys(a['action'])
        ua = a['action']
        ua['type'] = int(ua['type'])

        if 'parameter' in ua.keys():
            ua['parameter'] = int(ua['parameter'])
        if 'x' in ua.keys():
            ua['x'], ua['y'] = int(ua['x']), int(ua['y'])

    gs['time'] = int(gs['time'])
    pgs['width'] = int(pgs['width'])
    pgs['height'] = int(pgs['height'])

    # input()
    # gs = json.loads(json.dumps(gs))
    # print(gs)

    # TODO: this costs a lot time!
    ans = from_dict(data_class=GameState, data=gs)
    # print(odict)

    # print(time.time()  - st)
    return ans


def load_zips(path):
    files_name = [f for f in listdir(path)]
    return files_name


def store(records, path):
    """
    store records to disk
    :param path:
    :param records:
    :return:
    """

    with open(path, 'wb') as f:
        records = Records(records)
        dill.dump(records, f, recurse=True, protocol=dill.HIGHEST_PROTOCOL)
    # f = open("/home/toby/rcds.pck", 'rb')
    # rcd = dill.load(f)
    # print(rcd.records.__len__())


def main():
    # TODO: arg parse
    start = time.time()
    gss = []
    cnt = 0
    for f_n in load_zips(os.path.expanduser(tournament_dir)):
        with ZipFile(os.path.join(os.path.expanduser(tournament_dir), f_n)) as myzip:
            with myzip.open('game.xml') as myfile:
                cnt += 1
                if cnt % 100 == 0:
                    print('Game No.{} processed'.format(cnt))
                for x in parse_xml(myfile.read()):
                    gss.append(parse_gs(x))

    records = []
    for gs in gss:
        records.append(extract_record(gs, 1))
    end = time.time()
    print('total records:', len(records))
    print('total time:', end - start)

    # test
    # """
    # f = open("/home/toby/text", 'wb')
    # dill.dump(records[0], f, recurse=True)
    # f.close()
    # f = open("/home/toby/text", 'rb')
    # x = dill.load(f)
    # f.close()
    # assert x == records[0]
    # """
    store(records, os.path.expanduser(saving_dir))


if __name__ == '__main__':

    # pass
    main()
