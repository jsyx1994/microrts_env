import socket


class Player(object):
    conn = None
    type = None
    port = None
    brain = None
    client_ip = None
    id = None

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
        :param client_ip:
        :param port:
        :return:
        """
        server_socket = socket.socket()
        server_socket.bind((self.client_ip, self.port))
        server_socket.listen(5)
        print("Wait for Java client connection...")
        self.conn, address_info = server_socket.accept()

        print("Server: Send welcome msg to client...")
        print(self._send_msg("Welcome msg sent!"))

    def _send_msg(self, msg: str):
        try:
            self.conn.send(('%s\n' % msg).encode('utf-8'))
        except Exception as err:
            print("An error has occured: ", err)
        return self.conn.recv(65536).decode('utf-8')