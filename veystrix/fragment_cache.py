"""Tokenized prompt-fragment cache (blueprint UCH-08 / MIC-06).

A client resends the whole conversation on every tool turn, so the same
rendered prompt text is encoded again and again. The tokenizer is
deterministic, so the id list for an exact text is a pure function of
(tokenizer identity, text). This cache stores that mapping in a SQLite WAL
database keyed by a ``blake2b-256`` of both, bounded by entries and bytes, and
falls back to encoding on any error: a cache miss or a corrupt row can never
change a request's tokens.

It is opt-in with ``VEYSTRIX_FRAGMENT_CACHE=1``. The cache is model-scoped by
its key, so a Qwen and a Nemotron tokenizer never share an entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np

from veystrix.checkpoints import tokenizer_fingerprint

MAX_ENTRIES = 2048
MAX_BYTES = 64 * 2**20


def _identity_hash(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:32]


def identity_for_backend(backend, model_path=None) -> dict:
    """Everything that changes how a string becomes ids."""
    tokenizer = getattr(backend, "tokenizer", None)
    vocab = None
    try:
        vocab = len(tokenizer) if tokenizer is not None else None
    except TypeError:
        vocab = None
    path = model_path or getattr(backend, "model_path", None)
    source = getattr(tokenizer, "backend_tokenizer", None)
    serialize = getattr(source, "to_str", None)
    serialized = serialize() if callable(serialize) else None
    content = hashlib.sha256(serialized.encode()).hexdigest() if serialized is not None else None
    if content is None and path:
        content = tokenizer_fingerprint(Path(path).expanduser())
    return {"runtime": type(backend).__name__, "model": str(path or ""),
            "tokenizer": str(getattr(tokenizer, "name_or_path", "") or ""), "vocab": vocab,
            "content": content, "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
            "encode_format": 1}


def directory_for_cache(root=None) -> Path:
    from veystrix import universal_cache

    return Path(root) if root else universal_cache.cache_root() / "fragments-v1"


class TokenizedFragmentCache:
    def __init__(self, directory, identity: dict, *, max_entries: int = MAX_ENTRIES,
                 max_bytes: int = MAX_BYTES):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("fragment cache directory must be private and owned by this user")
        self.identity = _identity_hash(identity)
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1, int(max_bytes))
        self.path = self.directory / "fragments-v1.sqlite"
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS fragments "
                         "(k TEXT PRIMARY KEY, ids BLOB NOT NULL, n INTEGER NOT NULL, created REAL NOT NULL)")
        self._db.commit()
        self.hits = self.misses = 0
        self.stores = self.evictions = 0
        self.bytes_stored = 0

    def _key(self, text: str) -> str:
        return hashlib.blake2b((self.identity + "\0" + text).encode(), digest_size=32).hexdigest()

    def get(self, text: str) -> list[int] | None:
        key = self._key(text)
        try:
            with self._lock:
                row = self._db.execute("SELECT ids, n FROM fragments WHERE k = ?", (key,)).fetchone()
        except sqlite3.Error:
            return None
        if row is None or not isinstance(row[0], bytes) or len(row[0]) != row[1] or len(row[0]) % 4:
            self.misses += 1
            return None
        self.hits += 1
        return np.frombuffer(row[0], dtype="<u4").astype(np.int64).tolist()

    def put(self, text: str, ids) -> None:
        data = np.asarray(ids, dtype="<u4").tobytes()
        if not data:
            return
        if len(data) > self.max_bytes:
            return
        key = self._key(text)
        with self._lock:
            try:
                self._db.execute("INSERT OR REPLACE INTO fragments (k, ids, n, created) VALUES (?, ?, ?, ?)",
                                 (key, data, len(data), time.time()))
                self.bytes_stored += len(data)
                self.stores += 1
                self._evict_locked()
                self._db.commit()
            except sqlite3.Error:
                self._db.rollback()

    def _evict_locked(self) -> None:
        rows = self._db.execute("SELECT k, n FROM fragments ORDER BY created").fetchall()
        total = sum(row[1] for row in rows)
        count = len(rows)
        for key, size in rows:
            if count <= self.max_entries and total <= self.max_bytes:
                break
            self._db.execute("DELETE FROM fragments WHERE k = ?", (key,))
            total -= size
            count -= 1
            self.evictions += 1

    def stats(self) -> dict:
        with self._lock:
            try:
                entries, total = self._db.execute("SELECT COUNT(*), COALESCE(SUM(n), 0) FROM fragments").fetchone()
            except sqlite3.Error:
                entries, total = 0, 0
        lookups = self.hits + self.misses
        return {"enabled": True, "identity": self.identity, "entries": entries, "bytes": total,
                "max_entries": self.max_entries, "max_bytes": self.max_bytes,
                "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / lookups, 3) if lookups else None,
                "stores": self.stores, "evictions": self.evictions}

    def close(self) -> None:
        with self._lock:
            try:
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._db.close()
            except sqlite3.Error:
                pass


def open_for_backend(backend, root=None, **options) -> TokenizedFragmentCache | None:
    """Open the cache, or return None when it cannot be opened. Never fail a request over telemetry."""
    try:
        identity = identity_for_backend(backend)
        if identity["content"] is None:
            return None
        return TokenizedFragmentCache(directory_for_cache(root), identity, **options)
    except (OSError, ValueError, sqlite3.Error):
        return None
