"""Verify minisign signatures with the standard library only (no `minisign` binary needed).

Release archives are listed in SHA256SUMS.txt, and that file is signed with the owner's
minisign key. The updater checks the signature against the public key compiled into this
module, then checks each archive's SHA-256 against the signed list. Ed25519 verification
follows RFC 8032 section 6 (reference arithmetic; one verification takes well under a second).

minisign formats (https://jedisct1.github.io/minisign/):
  public key   base64( "Ed" | key_id[8] | public_key[32] )
  signature    untrusted comment line
               base64( "Ed"|"ED" | key_id[8] | signature[64] )   ("ED" = BLAKE2b-512 prehashed)
               "trusted comment: " + text
               base64( global_signature[64] )   over signature[64] + text
"""
from __future__ import annotations

import base64
import hashlib

# The Veystrix release key (public half). The private key never leaves the owner's Mac.
RELEASE_PUBLIC_KEY = "RWRqvDsIRe4F5ZmA+iMcMf+cnRjiG+k6NDEQFm7m0BsyVNmP/p9IUceg"

_P = 2 ** 255 - 19
_Q = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * _inv(_D * y * y + 1) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


def _add(a, b):
    ea = (a[1] - a[0]) * (b[1] - b[0]) % _P
    eb = (a[1] + a[0]) * (b[1] + b[0]) % _P
    ec = 2 * a[3] * b[3] * _D % _P
    ed = 2 * a[2] * b[2] % _P
    e, f, g, h = eb - ea, ed - ec, ed + ec, eb + ea
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _mul(scalar: int, point):
    result = (0, 1, 1, 0)
    while scalar:
        if scalar & 1:
            result = _add(result, point)
        point = _add(point, point)
        scalar >>= 1
    return result


def _equal(a, b) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


_GY = 4 * _inv(5) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)


def _decompress(raw: bytes):
    if len(raw) != 32:
        return None
    y = int.from_bytes(raw, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def ed25519_verify(public: bytes, message: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    a, r = _decompress(public), _decompress(signature[:32])
    if a is None or r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _Q:
        return False
    h = int.from_bytes(hashlib.sha512(signature[:32] + public + message).digest(), "little") % _Q
    return _equal(_mul(s, _G), _add(r, _mul(h, a)))


class SignatureError(ValueError):
    pass


def _b64(text: str, size: int, what: str) -> bytes:
    try:
        raw = base64.b64decode(text.strip(), validate=True)
    except ValueError as exc:
        raise SignatureError(f"{what} is not valid base64") from exc
    if len(raw) != size:
        raise SignatureError(f"{what} has the wrong length")
    return raw


def verify_minisign(data: bytes, signature_text: str, public_key: str = RELEASE_PUBLIC_KEY) -> str:
    """Return the trusted comment when ``signature_text`` is a valid signature of ``data`` by
    ``public_key``; raise SignatureError otherwise (never returns on a bad signature)."""
    key_line = [line for line in public_key.strip().splitlines() if not line.startswith("untrusted comment:")]
    key = _b64(key_line[-1] if key_line else "", 42, "public key")
    if key[:2] != b"Ed":
        raise SignatureError("unsupported public key algorithm")
    lines = [line for line in signature_text.strip().splitlines() if line.strip()]
    if len(lines) < 4 or not lines[2].startswith("trusted comment: "):
        raise SignatureError("malformed signature file")
    sig = _b64(lines[1], 74, "signature")
    algorithm, key_id, signature = sig[:2], sig[2:10], sig[10:]
    if key_id != key[2:10]:
        raise SignatureError("signed with a different key")
    if algorithm == b"ED":
        message = hashlib.blake2b(data, digest_size=64).digest()
    elif algorithm == b"Ed":
        message = data
    else:
        raise SignatureError("unsupported signature algorithm")
    if not ed25519_verify(key[10:], message, signature):
        raise SignatureError("signature does not match the file")
    trusted = lines[2][len("trusted comment: "):]
    if not ed25519_verify(key[10:], signature + trusted.encode("utf-8"),
                          _b64(lines[3], 64, "global signature")):
        raise SignatureError("trusted comment was altered")
    return trusted
