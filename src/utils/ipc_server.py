import json
import os
import socket
import struct
import threading
from collections import deque

SOCKET_STATE = "/tmp/neuracam.sock"
SOCKET_INPUT = "/tmp/neuracam_input.sock"
MAX_EVENTS = 30
MSG_TYPE_STATE = 0
MSG_TYPE_FRAME = 1
MSG_TYPE_KEY = 2


def _pack_msg(msg_type: int, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + bytes([msg_type]) + payload


class IPCServer:
    def __init__(self, state_sock=SOCKET_STATE, input_sock=SOCKET_INPUT):
        self.state_sock_path = state_sock
        self.input_sock_path = input_sock
        self._state_server = None
        self._input_server = None
        self._state_clients = []
        self._lock = threading.Lock()
        self._events = deque(maxlen=MAX_EVENTS)
        self._key_buffer = deque()
        self._running = False
        self._start()

    def add_event(self, msg: str):
        from datetime import datetime
        entry = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        with self._lock:
            self._events.append(entry)

    def read_key(self):
        with self._lock:
            if self._key_buffer:
                return self._key_buffer.popleft()
        return None

    def _start(self):
        for p in [self.state_sock_path, self.input_sock_path]:
            try:
                os.unlink(p)
            except OSError:
                pass

        self._state_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._state_server.bind(self.state_sock_path)
        self._state_server.listen(5)
        self._state_server.setblocking(False)

        self._input_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._input_server.bind(self.input_sock_path)
        self._input_server.listen(5)
        self._input_server.setblocking(False)

        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self):
        import select
        while self._running:
            try:
                readable, _, _ = select.select(
                    [self._state_server, self._input_server], [], [], 0.1
                )
                for s in readable:
                    if s == self._state_server:
                        client, _ = s.accept()
                        with self._lock:
                            self._state_clients.append(client)
                    elif s == self._input_server:
                        client, _ = s.accept()
                        t = threading.Thread(
                            target=self._read_input_loop, args=(client,), daemon=True
                        )
                        t.start()
                        with self._lock:
                            pass
            except (BlockingIOError, OSError):
                pass

    def _read_input_loop(self, client: socket.socket):
        try:
            while self._running:
                data = client.recv(1024)
                if not data:
                    break
                for c in data.decode("utf-8", errors="replace"):
                    with self._lock:
                        self._key_buffer.append(c)
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                client.close()
            except OSError:
                pass

    def send_state(self, state: dict, frame_jpeg: bytes = None):
        state_bytes = json.dumps(state).encode("utf-8")
        state_msg = _pack_msg(MSG_TYPE_STATE, state_bytes)
        dead = []
        with self._lock:
            for client in self._state_clients:
                try:
                    client.sendall(state_msg)
                    if frame_jpeg:
                        client.sendall(_pack_msg(MSG_TYPE_FRAME, frame_jpeg))
                except (BrokenPipeError, ConnectionError, OSError):
                    dead.append(client)
            for c in dead:
                self._state_clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    def stop(self):
        self._running = False
        with self._lock:
            for c in self._state_clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._state_clients.clear()
        for s in [self._state_server, self._input_server]:
            if s:
                try:
                    s.close()
                except OSError:
                    pass
        for p in [self.state_sock_path, self.input_sock_path]:
            try:
                os.unlink(p)
            except OSError:
                pass
