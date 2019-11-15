import socket
from dataclasses import asdict
from typing import List, Any
from rts_wrapper.datatypes import *
import json

def get_available_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port




