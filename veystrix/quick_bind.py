"""HTTP servers that start without a reverse DNS lookup.

``http.server.HTTPServer.server_bind`` calls ``socket.getfqdn(host)`` only to fill in ``server_name``, which nothing here
uses. On a machine whose resolver is slow the first call can take tens of seconds (35 s on a GitHub macOS runner, where it
made every freshly started engine, router and test helper look hung), and it adds nothing for a loopback address.
"""
import socketserver


class QuickBindMixin:
    """Put before ``ThreadingHTTPServer`` / ``HTTPServer`` in the bases: binds like ``TCPServer`` and skips ``getfqdn``."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port
