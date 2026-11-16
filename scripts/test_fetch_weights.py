"""Check that a model download survives a dropped connection. The server runs
on this machine only; nothing leaves it. Run: python3 -m pytest scripts"""

import http.server
import threading

from fetch_weights import download

PAYLOAD = bytes(range(256)) * 400


class DroppingServer(http.server.BaseHTTPRequestHandler):
    """Serves PAYLOAD, honours Range, and cuts the first response short."""

    requests = 0

    def do_GET(self):
        DroppingServer.requests += 1
        wanted = self.headers.get("Range")
        start = int(wanted.split("=")[1].split("-")[0]) if wanted else 0
        body = PAYLOAD[start:]
        self.send_response(206 if wanted else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if DroppingServer.requests == 1:
            body = body[:len(body) // 3]
        self.wfile.write(body)

    def log_message(self, *arguments):
        pass


def test_download_continues_after_a_dropped_connection(tmp_path):
    DroppingServer.requests = 0
    server = http.server.HTTPServer(("127.0.0.1", 0), DroppingServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        address = f"http://127.0.0.1:{server.server_port}/file"
        partial = download(address, tmp_path / "file.bin", len(PAYLOAD), wait=0)
    finally:
        server.shutdown()
        server.server_close()
    assert partial.read_bytes() == PAYLOAD
    assert DroppingServer.requests == 2
