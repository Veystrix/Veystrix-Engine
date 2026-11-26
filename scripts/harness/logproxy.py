"""Logging reverse proxy: logproxy.py LISTEN_PORT UPSTREAM_PORT OUT.jsonl
Forwards every request to 127.0.0.1:UPSTREAM unchanged (streams the response back) and appends
{"t", "method", "path", "body"} per request to OUT.jsonl."""
import http.client, http.server, json, socketserver, sys, time

LISTEN, UP, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _fwd(self, method):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        with open(OUT, "a") as f:
            try:
                parsed = json.loads(body) if body else None
            except ValueError:
                parsed = body.decode("utf-8", "replace")
            f.write(json.dumps({"t": time.time(), "method": method, "path": self.path, "body": parsed}) + "\n")
        conn = http.client.HTTPConnection("127.0.0.1", UP, timeout=7200)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        conn.request(method, self.path, body=body, headers=headers)
        r = conn.getresponse()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.close_connection = True

    def do_GET(self):
        self._fwd("GET")

    def do_POST(self):
        self._fwd("POST")

    def log_message(self, *a):
        pass


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


S(("127.0.0.1", LISTEN), H).serve_forever()
