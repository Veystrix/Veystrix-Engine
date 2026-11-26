"""Versioned, 64 KiB-aligned records for affine routed expert weights."""
from __future__ import annotations

from contextlib import ExitStack
import ctypes
import hashlib
import fcntl
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import tempfile

ALIGNMENT = 64 * 1024
RESIDENT_SAMPLES = 4      # pages probed per record, spread evenly, plus the last one
_LIBC = ctypes.CDLL(None, use_errno=True)
HEADER = struct.Struct("<8sI")
MAGIC = b"VXEXPK01"
_PATTERNS = {
    "flashnext": r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.(.+)",
    "nemotron35": r"backbone\.layers\.(\d+)\.mixer\.switch_mlp\.(.+)",
}
_PROJECTIONS = {"flashnext": ("gate_proj", "up_proj", "down_proj"), "nemotron35": ("fc1", "fc2")}


def source_layout(directory, index, runtime):
    """Derive the record layout and identity from the original shard headers."""
    parts = tuple(f"{projection}.{part}" for projection in _PROJECTIONS[runtime]
                  for part in ("weight", "scales", "biases"))
    layers = {}
    for name, desc in index.items():
        match = re.fullmatch(_PATTERNS[runtime], name)
        if match and match[2] in parts:
            layers.setdefault(int(match[1]), {})[match[2]] = desc
    if not layers or any(layer.keys() != set(parts) for layer in layers.values()):
        raise ValueError("incomplete routed expert inventory")
    ordered = {layer: [layers[layer][part] for part in parts] for layer in sorted(layers)}
    first = next(iter(ordered.values()))
    geometry = [(d["dtype"], d["shape"], d["length"]) for d in first]
    if any(len(shape) != 3 for _, shape, _ in geometry):
        raise ValueError("routed expert tensors must have three dimensions")
    count = first[0]["shape"][0]
    if count <= 0 or any(shape[0] != count or length <= 0 or length % count for _, shape, length in geometry):
        raise ValueError("unsupported routed expert geometry")
    if any([(d["dtype"], d["shape"], d["length"]) for d in row] != geometry for row in ordered.values()):
        raise ValueError("routed expert geometry varies by layer")
    sizes = [length // count for _, _, length in geometry]
    payload = sum(sizes)
    shard_names = sorted({desc["shard"] for row in ordered.values() for desc in row})
    file_identity = {}
    for name in shard_names:
        info = (Path(directory) / name).stat()
        file_identity[name] = (info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev)
    source = {"config": hashlib.sha256((Path(directory) / "config.json").read_bytes()).hexdigest(),
              "tensors": ordered, "shards": file_identity}
    identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return ordered, {
        "runtime": runtime, "identity": identity, "expert_count": count,
        "parts": list(parts), "sizes": sizes, "payload_bytes": payload,
        "stride": (payload + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT,
    }


DEFAULT_DIRECTORY = "~/.cache/veystrix/expert-packs"
DIRECTORY_ENV = "VEYSTRIX_EXPERT_PACK_DIR"


def default_path(directory, index, runtime):
    """Where the full pack of this checkpoint is kept: one file per source identity.

    The identity covers the config and every shard's size, mtime and inode, so a
    changed or replaced checkpoint never finds an older pack."""
    _, layout = source_layout(directory, index, runtime)
    base = Path(os.environ.get(DIRECTORY_ENV) or DEFAULT_DIRECTORY).expanduser()
    return base / f"{runtime}-{layout['identity'][:16]}.pack"


def build(directory, index, runtime, destination, *, layers=None, content_addressed=False):
    """Pack selected layers and publish the completed file atomically."""
    directory, destination = Path(directory), Path(destination)
    source, layout = source_layout(directory, index, runtime)
    layers = sorted(source if layers is None else set(layers))
    if not layers or any(layer not in source for layer in layers):
        raise ValueError("select at least one valid routed layer")
    if not content_addressed and (destination.exists() or destination.is_symlink()):
        raise FileExistsError(destination)
    parent_dir = destination if content_addressed else destination.parent
    layout["layers"] = layers
    count, stride = layout["expert_count"], layout["stride"]
    size = ALIGNMENT + len(layers) * count * stride
    if shutil.disk_usage(parent_dir).free < size:
        raise OSError(f"expert pack requires {size:,} bytes of free disk")
    metadata = json.dumps(layout, separators=(",", ":")).encode()
    if len(metadata) > ALIGNMENT - HEADER.size:
        raise ValueError("expert pack header exceeds alignment")
    with ExitStack() as cleanup:
        fd, temporary = tempfile.mkstemp(prefix=".expert-pack-", dir=parent_dir)
        cleanup.callback(os.unlink, temporary)
        handle = cleanup.enter_context(os.fdopen(fd, "wb"))
        digest = hashlib.sha256() if content_addressed else None

        def write(data):
            handle.write(data)
            if digest is not None:
                digest.update(data)

        header = HEADER.pack(MAGIC, len(metadata)) + metadata
        write(header.ljust(ALIGNMENT, b"\0"))
        fds = {}
        padding = bytes(stride - layout["payload_bytes"])
        for layer in layers:
            for expert in range(count):
                for desc, length in zip(source[layer], layout["sizes"], strict=True):
                    shard = desc["shard"]
                    if shard not in fds:
                        fds[shard] = os.open(directory / shard, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                        cleanup.callback(os.close, fds[shard])
                    data = os.pread(fds[shard], length, desc["offset"] + expert * length)
                    if len(data) != length:
                        raise OSError(f"short source read for layer {layer}, expert {expert}")
                    write(data)
                write(padding)
        handle.flush()
        os.fsync(handle.fileno())
        _, final_layout = source_layout(directory, index, runtime)
        if final_layout["identity"] != layout["identity"]:
            raise OSError("source checkpoint changed during expert packing")
        if digest is not None:
            destination = destination / (digest.hexdigest() + ".pack")
        os.link(temporary, destination)  # atomic publication without replacing a concurrent build
        parent = os.open(parent_dir, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    return (destination, size) if content_addressed else size


class ExpertPack:
    """Read into caller-owned buffers through one positioned read per expert."""

    def __init__(self, path, directory, index, runtime, *, uncached=False, mapped=False):
        source, layout = source_layout(directory, index, runtime)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("expert pack must be a regular file")
            prefix = os.pread(fd, HEADER.size, 0)
            if len(prefix) != HEADER.size:
                raise ValueError("truncated expert pack header")
            magic, size = HEADER.unpack(prefix)
            if magic != MAGIC or not 0 < size <= ALIGNMENT - HEADER.size:
                raise ValueError("invalid expert pack header")
            loaded = json.loads(os.pread(fd, size, HEADER.size))
            if not isinstance(loaded, dict) or any(loaded.get(key) != value for key, value in layout.items()):
                raise ValueError("expert pack does not match the source checkpoint")
            layers = loaded.get("layers")
            if (not isinstance(layers, list) or not layers
                    or any(type(layer) is not int or layer not in source for layer in layers)
                    or sorted(set(layers)) != layers):
                raise ValueError("invalid packed layer list")
            self.count, self.stride = layout["expert_count"], layout["stride"]
            self.payload = layout["payload_bytes"]
            self.parts = dict(zip(layout["parts"], layout["sizes"], strict=True))
            if info.st_size != ALIGNMENT + len(layers) * self.count * self.stride:
                raise ValueError("invalid expert pack size")
            self.layers = {layer: i for i, layer in enumerate(layers)}
            if uncached:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            self.mapping = mmap.mmap(fd, 0, access=mmap.ACCESS_READ) if mapped else None
            self._base = None
            if self.mapping is not None:
                import numpy

                # Only the address is kept; the one-byte view is released at the end of the statement.
                self._base = numpy.frombuffer(self.mapping, dtype=numpy.uint8, count=1).ctypes.data
                pages = -(-self.payload // mmap.PAGESIZE)
                step = max(1, pages // RESIDENT_SAMPLES)
                self._samples = [index * mmap.PAGESIZE for index in sorted({*range(0, pages, step), pages - 1})]
                self._probe = (ctypes.c_char * 1)()
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd

    def offset(self, layer, expert):
        if not 0 <= expert < self.count:
            raise ValueError("invalid packed expert index")
        return ALIGNMENT + (self.layers[layer] * self.count + expert) * self.stride

    def resident(self, layer, expert):
        """Whether the kernel holds this record in memory, from a sample of its pages (mapped packs only).

        Every page of a record is read and used together, so the pages age together and a sample of
        RESIDENT_SAMPLES pages spread over the record is enough. A wrong answer costs time, never
        correctness: a missed cold page is faulted in by whoever touches it, and a record wrongly called cold
        is read from the file cache. A full mincore over a 5.6 MB record costs 80 us; the sample about 4 us."""
        if self._base is None:
            return False
        start = self._base + self.offset(layer, expert)
        for position in self._samples:
            if _LIBC.mincore(ctypes.c_void_p(start + position), ctypes.c_size_t(1), self._probe) != 0:
                return False
            if not self._probe[0][0] & 1:
                return False
        return True

    def read_into(self, layer, expert, buffers, *, read=os.preadv):
        if not 0 <= expert < self.count or sum(buffer.nbytes for buffer in buffers) != self.payload:
            raise ValueError("invalid expert pack read")
        if read(self.fd, buffers, self.offset(layer, expert)) != self.payload:
            raise OSError(f"short packed read for layer {layer}, expert {expert}")

    def read_extent(self, layer, first, buffers, *, read=os.preadv):
        """Read consecutive aligned records into distinct payload and padding vectors."""
        size = sum(buffer.nbytes for buffer in buffers)
        if not size or size % self.stride or first + size // self.stride > self.count:
            raise ValueError("invalid expert pack extent")
        offset = self.offset(layer, first)
        if self.mapping is not None:
            with memoryview(self.mapping) as source:
                for buffer in buffers:
                    buffer[:] = source[offset:offset + buffer.nbytes]
                    offset += buffer.nbytes
        elif read(self.fd, buffers, offset) != size:
            raise OSError(f"short packed extent for layer {layer}, expert {first}")
        return size

    def close(self):
        if self.mapping is not None:
            try:
                self.mapping.close()
            except BufferError:
                pass  # zero-copy arrays still view the mapping; it is unmapped when the last one is released
            self.mapping = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
