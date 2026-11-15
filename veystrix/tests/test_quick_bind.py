"""The engine, the router and the test helpers start without a reverse DNS lookup (it took 35 s on the CI runner)."""
import socket
import unittest
from unittest.mock import patch

from veystrix.model_router import RouterServer
from veystrix.quick_bind import QuickBindMixin
from veystrix.server import VeystrixHTTPServer


def no_lookup(*_args, **_kwargs):
    raise AssertionError("socket.getfqdn was called while binding a server")


class QuickBindTests(unittest.TestCase):
    def test_servers_do_not_resolve_their_own_address(self):
        with patch.object(socket, "getfqdn", no_lookup):
            for make in (lambda: VeystrixHTTPServer(("127.0.0.1", 0), None),
                         lambda: RouterServer(("127.0.0.1", 0), None, None)):
                server = make()
                try:
                    self.assertEqual(server.server_name, "127.0.0.1")
                    self.assertEqual(server.server_port, server.server_address[1])
                    self.assertGreater(server.server_port, 0)
                finally:
                    server.server_close()

    def test_the_standard_server_would_have_called_it(self):
        """Guards the premise: without the mixin, the standard library asks for the fully qualified name."""
        from http.server import ThreadingHTTPServer
        with patch.object(socket, "getfqdn", no_lookup):
            with self.assertRaises(AssertionError):
                ThreadingHTTPServer(("127.0.0.1", 0), object)

    def test_mixin_keeps_address_reuse_and_binds(self):
        from http.server import ThreadingHTTPServer

        class Plain(QuickBindMixin, ThreadingHTTPServer):
            pass
        server = Plain(("127.0.0.1", 0), object)
        try:
            self.assertTrue(server.allow_reuse_address)
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
