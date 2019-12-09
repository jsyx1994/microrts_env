import socket

from rts_wrapper.envs.utils import signal_wrapper, pa_to_jsonable


class Player(object):
    conn = None
    type = None
    port = None
    brain = None
    client_ip = None
    id = None

    action = []

    def __str__(self):
        pass

    def __init__(self, pid, client_ip, port):
        self.id = pid
        self.port = port
        self.client_ip = client_ip

    def load_brain(self):
        raise NotImplementedError

    def join(self):
        """
        hand shake with java end
        """
        server_socket = socket.socket()
        server_socket.bind((self.client_ip, self.port))
        server_socket.listen(5)
        print("Player{} Wait for Java client connection...".format(self.id))
        self.conn, address_info = server_socket.accept()

        self.greetings()

    def greetings(self):
        print("Player{}: Send welcome msg to client...".format(self.id))
        print(self._send_msg("Welcome msg sent!"))

    def reset(self):
        print("Server: Send reset command...")
        raw = self._send_msg('reset')
        # print(raw)
        return signal_wrapper(raw)

    def think(self, helper, **kwargs):
        """
        :param helper: the function
        figure out a solution according to obs
        """
        self.action.clear()

    def act(self, action):
        pa = pa_to_jsonable(action)
        raw = self._send_msg(pa)
        return signal_wrapper(raw)

    def _send_msg(self, msg: str):
        try:
            self.conn.send(('%s\n' % msg).encode('utf-8'))
        except Exception as err:
            print("An error has occured: ", err)
        return self.conn.recv(65536).decode('utf-8')