"""Runtime bytecode decryption bootstrap for packaged engine builds.

This module is intentionally small and readable: it contains no product logic.
It installs an import hook that decrypts ``*.pyc.enc`` files using a 256-bit key
compiled into ``_protect_key.dylib`` at build time. Source checkouts never use
it. This raises the cost of casual decompilation; it is NOT unbreakable DRM —
anyone who controls the machine can call the same dylib or patch the hook.
"""
from __future__ import annotations

import ctypes
import hashlib
import hmac
import importlib.abc
import importlib.machinery
import marshal
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KEY_DYLIB = _HERE / "_protect_key.dylib"
_NONCE_LEN = 16
_TAG_LEN = 32
_HEADER_LEN = 16  # Python 3.12 .pyc header
_INSTALLED = False


def key() -> bytes:
    lib = ctypes.CDLL(str(_KEY_DYLIB))
    lib.vx_protect_key.restype = ctypes.c_void_p
    length = lib.vx_protect_key_length()
    return ctypes.string_at(lib.vx_protect_key(), length)


def _keystream(key: bytes, nonce: bytes, size: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out.extend(hmac.new(key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(out[:size])


def encrypt(plaintext: bytes, secret: bytes, nonce: bytes | None = None) -> bytes:
    nonce = nonce or hashlib.sha256(plaintext).digest()[:_NONCE_LEN]
    stream = _keystream(secret, nonce, len(plaintext))
    encrypted = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag = hmac.new(secret, nonce + encrypted, hashlib.sha256).digest()
    return nonce + encrypted + tag


def decrypt(blob: bytes, secret: bytes | None = None) -> bytes:
    secret = secret or key()
    if len(blob) < _NONCE_LEN + _TAG_LEN:
        raise ValueError("encrypted bytecode is truncated")
    nonce, encrypted, tag = blob[:_NONCE_LEN], blob[_NONCE_LEN:-_TAG_LEN], blob[-_TAG_LEN:]
    if not hmac.compare_digest(hmac.new(secret, nonce + encrypted, hashlib.sha256).digest(), tag):
        raise ValueError("encrypted bytecode failed authentication")
    stream = _keystream(secret, nonce, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, stream))


def load_code(path: Path):
    pyc = decrypt(Path(path).read_bytes())
    return marshal.loads(pyc[_HEADER_LEN:])


class _EncryptedLoader(importlib.abc.Loader):
    def __init__(self, origin: Path):
        self.origin = origin

    def get_filename(self, fullname):
        return str(self.origin)

    def get_code(self, fullname):
        return load_code(self.origin)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.origin)
        exec(self.get_code(module.__name__), module.__dict__)


class _EncryptedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        leaf = fullname.rsplit(".", 1)[-1]
        roots = list(path) if path is not None else list(sys.path)
        for root in roots:
            base = Path(root)
            package = base / leaf / "__init__.pyc.enc"
            if package.is_file():
                spec = importlib.machinery.ModuleSpec(
                    fullname, _EncryptedLoader(package), origin=str(package), is_package=True)
                spec.submodule_search_locations = [str(base / leaf)]
                return spec
            module = base / f"{leaf}.pyc.enc"
            if module.is_file():
                return importlib.machinery.ModuleSpec(
                    fullname, _EncryptedLoader(module), origin=str(module))
        return None


def install() -> None:
    global _INSTALLED
    if not _INSTALLED:
        sys.meta_path.insert(0, _EncryptedFinder())
        _INSTALLED = True
