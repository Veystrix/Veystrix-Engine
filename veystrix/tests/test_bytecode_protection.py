"""Bytecode protection must round-trip and reject tampering."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_protect():
    path = ROOT / "packaging" / "engine" / "_protect.py"
    spec = importlib.util.spec_from_file_location("_protect_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BytecodeProtectionTests(unittest.TestCase):
    def setUp(self):
        self.protect = _load_protect()
        self.secret = b"k" * 32

    def test_round_trip(self):
        payload = b"compiled bytecode payload" * 10
        blob = self.protect.encrypt(payload, self.secret)
        self.assertNotIn(b"compiled bytecode", blob)
        self.assertEqual(self.protect.decrypt(blob, self.secret), payload)

    def test_tampering_is_rejected(self):
        blob = bytearray(self.protect.encrypt(b"secret", self.secret))
        blob[len(blob) // 2] ^= 0x01
        with self.assertRaises(ValueError):
            self.protect.decrypt(bytes(blob), self.secret)

    def test_wrong_key_is_rejected(self):
        blob = self.protect.encrypt(b"secret", self.secret)
        with self.assertRaises(ValueError):
            self.protect.decrypt(blob, b"x" * 32)

    def test_truncated_blob_is_rejected(self):
        with self.assertRaises(ValueError):
            self.protect.decrypt(b"short", self.secret)


if __name__ == "__main__":
    unittest.main()
