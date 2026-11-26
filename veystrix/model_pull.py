"""``veystrix models pull <model>``: download a supported checkpoint, safely.

* Only known families (``model_identity.IDENTITIES`` with a ``download_repo``) or an explicit
  ``owner/repo`` are accepted; the destination is ``<model root>/<checkpoint directory>``.
* Disk space is checked before anything is fetched (remaining bytes + a safety margin).
* Resumable: files already present and verified are skipped; partial downloads resume
  (huggingface_hub keeps ``.incomplete`` files under ``<dest>/.cache``).
* Integrity: every LFS file is checked against the SHA-256 the Hub publishes, small files by
  size. A file that exists with the WRONG content is never overwritten: the pull stops and
  says which file, so nothing the user already has is destroyed.
* After the download the inspect report and the family's structural check run (headers only).

The network/client and disk probes are injectable so the unit tests never download.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from veystrix import model_identity

MARGIN_BYTES = 2 * 1024**3
VERIFIED_NAME = ".veystrix-verified.json"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int
    sha256: str | None          # LFS files only


class PullError(RuntimeError):
    pass


def resolve_target(name: str, root: Path, allow_locked: bool = False) -> tuple[str, Path, model_identity.ModelIdentity | None]:
    """(repo, destination, identity) for an unlocked family name, alias or explicit owner/repo."""
    key = name.strip().lower()
    for identity in model_identity.IDENTITIES.values():
        matched = key in (identity.family, identity.checkpoint_alias, identity.served_id,
                          identity.download_repo.lower())
        if matched and identity.download_repo:
            if identity.locked and not allow_locked:
                raise PullError(f"{identity.display_name} ({identity.family}) is locked and unavailable in this build")
            folder = identity.checkpoint_directory or identity.download_repo.split("/")[-1]
            return identity.download_repo, root / folder, identity
    if "/" in name and not name.startswith("/") and ".." not in name:
        return name, root / name.split("/")[-1], None
    known = ", ".join(i.family for i in model_identity.IDENTITIES.values() if i.download_repo and not i.locked)
    raise PullError(f"unknown model {name!r}; use one of: {known}, or an explicit owner/repo")


class HubClient:
    """Thin wrapper over huggingface_hub (imported lazily)."""

    def list_files(self, repo: str) -> list[RemoteFile]:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        files = []
        for sibling in info.siblings or []:
            lfs = getattr(sibling, "lfs", None)
            sha = getattr(lfs, "sha256", None) if lfs is not None else None
            if sha is None and isinstance(lfs, dict):
                sha = lfs.get("sha256")
            files.append(RemoteFile(sibling.rfilename, int(sibling.size or 0), sha))
        return files

    def download(self, repo: str, filename: str, dest: Path) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo, filename, local_dir=str(dest)))


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_verified(dest: Path) -> dict:
    try:
        data = json.loads((dest / VERIFIED_NAME).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _verified(dest: Path, record: dict, remote: RemoteFile, hasher) -> bool | None:
    """True verified, False present-but-wrong, None absent."""
    path = dest / remote.name
    if not path.is_file():
        return None
    stat = path.stat()
    if stat.st_size != remote.size and remote.size:
        return False
    if remote.sha256 is None:
        return True
    cached = record.get(remote.name)
    if isinstance(cached, dict) and cached.get("size") == stat.st_size and \
            cached.get("mtime") == int(stat.st_mtime) and cached.get("sha256") == remote.sha256:
        return True
    ok = hasher(path) == remote.sha256
    if ok:
        record[remote.name] = {"size": stat.st_size, "mtime": int(stat.st_mtime), "sha256": remote.sha256}
    return ok


def pull(name: str, root: Path, *, client=None, free_bytes=None, hasher=None, echo=None,
         dry_run: bool = False, inspect_after: bool = True) -> dict:
    echo = echo or (lambda _m: None)
    client = client or HubClient()
    hasher = hasher or sha256_file
    repo, dest, identity = resolve_target(name, Path(root).expanduser())
    files = [f for f in client.list_files(repo) if not f.name.startswith(".")]
    if not files:
        raise PullError(f"{repo} lists no files")
    dest.mkdir(parents=True, exist_ok=True)
    record = _load_verified(dest)
    todo, done, wrong = [], [], []
    for remote in files:
        state = _verified(dest, record, remote, hasher)
        (done if state else wrong if state is False else todo).append(remote)
    if wrong:
        raise PullError("these files already exist in " + str(dest) + " with different content and were "
                        "left untouched: " + ", ".join(f.name for f in wrong)
                        + ". Move them aside yourself if you want a fresh download.")
    need = sum(f.size for f in todo)
    free = free_bytes(dest) if free_bytes else shutil.disk_usage(dest).free
    report = {"repo": repo, "destination": str(dest), "files": len(files), "already_verified": len(done),
              "to_download": len(todo), "bytes_to_download": need, "free_bytes": free}
    if todo and need + MARGIN_BYTES > free:
        raise PullError(f"not enough disk space: {need / 1024**3:.1f} GiB to download + "
                        f"{MARGIN_BYTES / 1024**3:.0f} GiB margin, {free / 1024**3:.1f} GiB free on {dest}")
    if dry_run:
        report["dry_run"] = True
        return report
    started = time.monotonic()
    # Small files first (config, tokenizer) so an interrupted pull is still inspectable.
    for remote in sorted(todo, key=lambda f: f.size):
        echo(f"downloading {remote.name} ({remote.size / 1024**2:,.0f} MiB)")
        client.download(repo, remote.name, dest)
        state = _verified(dest, record, remote, hasher)
        if not state:
            raise PullError(f"integrity check failed for {remote.name} (size or SHA-256 differs); "
                            "run the pull again to resume")
    (dest / VERIFIED_NAME).write_text(json.dumps(record, indent=1, sort_keys=True))
    report["seconds"] = round(time.monotonic() - started, 1)
    report["verified"] = True
    if inspect_after:
        from veystrix import model_inspect
        report["inspect"] = model_inspect.render(model_inspect.inspect(dest))
        if identity is not None and identity.family == "inkling2-mini":
            from veystrix.checkpoints import inkling2_mini_problems
            report["problems"] = inkling2_mini_problems(dest)
    return report
