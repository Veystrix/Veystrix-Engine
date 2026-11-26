"""Collect host and source identity before an arm starts."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import plistlib
import subprocess


def command(*args, cwd=None):
    try:
        return subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None


def output(*args, cwd=None):
    value = command(*args, cwd=cwd)
    return value.decode("utf-8", errors="replace").strip() if value is not None else None


def collect(root, checkpoint):
    commit = output("git", "rev-parse", "HEAD", cwd=root)
    diff = command("git", "diff", "--no-ext-diff", "--binary", "HEAD", "--", ".", ":(exclude)results/**", cwd=root)
    disk = command("diskutil", "info", "-plist", str(checkpoint))
    disk_stats = os.statvfs(checkpoint)
    try:
        info = plistlib.loads(disk) if disk else {}
    except (ValueError, plistlib.InvalidFileException):
        info = {}
    versions = {}
    for name in ("mlx", "mlx-lm", "mlx-vlm"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "code": {"commit": commit, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None},
        "dependencies": {"python": platform.python_version(), **versions},
        "host": {
            "platform": platform.platform(),
            "macos": platform.mac_ver()[0],
            "model": output("sysctl", "-n", "hw.model"),
            "soc": output("sysctl", "-n", "machdep.cpu.brand_string"),
            "ram_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
            "storage": {
                "device": info.get("DeviceIdentifier"),
                "media": info.get("MediaName"),
                "solid_state": info.get("SolidState"),
                "device_bytes": info.get("DiskSize"),
                "filesystem_bytes": disk_stats.f_blocks * disk_stats.f_frsize,
                "filesystem_free_bytes": disk_stats.f_bavail * disk_stats.f_frsize,
            },
            "power": output("pmset", "-g", "batt"),
            "thermal": output("pmset", "-g", "therm"),
        },
    }


def checkpoint_metadata(path):
    names = ("config.json", "generation_config.json", "model.safetensors.index.json",
             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    return {
        name: hashlib.sha256((Path(path) / name).read_bytes()).hexdigest()
        for name in names if (Path(path) / name).is_file()
    }
