import io
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional


class MJPEGHandler(BaseHTTPRequestHandler):
    frame: Optional[bytes] = None
    server_version = "NeuraCam/1.0"

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='margin:0;background:#000;'>"
                b"<img src='/stream' style='width:100vw;height:100vh;object-fit:contain;'>"
                b"</body></html>"
            )
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while not MJPEGServer._stop:
                frame = MJPEGHandler.frame
                if frame:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionError, OSError):
                        break
                else:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: text/plain\r\n\r\n")
                    self.wfile.write(b"waiting for frame...\r\n")
                threading.Event().wait(0.033)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a, **kw):
        pass


class MJPEGServer:
    _stop = False

    def __init__(self, port: int = 8080):
        self.port = port
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        MJPEGServer._stop = False
        found = False
        for p in range(self.port, self.port + 10):
            try:
                self._httpd = HTTPServer(("0.0.0.0", p), MJPEGHandler)
                self.port = p
                found = True
                break
            except (OSError, socket.error):
                continue
        if not found:
            print(f"MJPEG: no free port in range {self.port}-{self.port+9}")
            return
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f"MJPEG: http://localhost:{self.port}/")

    def publish(self, jpeg: bytes):
        MJPEGHandler.frame = jpeg

    def stop(self):
        MJPEGServer._stop = True
        if self._httpd:
            self._httpd.shutdown()
