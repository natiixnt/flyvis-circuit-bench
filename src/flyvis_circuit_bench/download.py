"""Download and verify the measurement package from a GitHub Release.

The asset URL is built from `--owner` (the GitHub account or organization that hosts the repository); nothing
is hard-coded. Verification has two levels: the archive against the release `SHA256SUMS`, then every extracted
file against the `SHA256SUMS` inside the archive. Extraction accepts regular files under the package directory
only (no links, no absolute paths, no `..`).
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from . import __version__
from .data import CHECKSUMS, read_checksums, verify

REPO = "flyvis-circuit-bench"
# Package 1.1.0 = package 1.0.0 byte for byte plus task/ and silence/ (Z4, Z6). Version 1.0.0 stays available.
DATA_VERSION = "1.1.0"
VERSIONS = ("1.0.0", "1.1.0")
PACKAGE = f"measurements-v{DATA_VERSION}"
ASSET = f"{PACKAGE}.tar.gz"


def asset_name(version: str = DATA_VERSION) -> str:
    if version not in VERSIONS:
        raise ValueError(f"unknown data version {version!r}; available: {', '.join(VERSIONS)}")
    return f"measurements-v{version}.tar.gz"
USER_AGENT = f"{REPO}/{__version__}"


def release_url(owner: str, name: str, repo: str = REPO, tag: str = f"v{DATA_VERSION}") -> str:
    if not owner or "/" in owner or owner.strip() != owner:
        raise ValueError("--owner must be a GitHub account or organization name")
    return f"https://github.com/{owner}/{repo}/releases/download/{tag}/{name}"


def _fetch(url: str, dest: Path, timeout: float = 60.0) -> None:
    if not url.startswith("https://"):
        raise ValueError("only https downloads are allowed")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive: Path, dest_parent: Path) -> Path:
    """Extract `archive` into `dest_parent/<package>`; refuse anything that is not a plain file inside it."""
    dest_parent = Path(dest_parent)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            parts = PurePosixPath(m.name).parts
            if not m.name or m.name.startswith("/") or ".." in parts or "\\" in m.name:
                raise ValueError(f"unsafe path in archive: {m.name!r}")
        roots = {PurePosixPath(m.name).parts[0] for m in members if m.name}
        if len(roots) != 1:
            raise ValueError("archive must contain exactly one top-level directory")
        root = roots.pop()
        target = dest_parent / root
        if target.exists():
            raise FileExistsError(f"{target} already exists")
        for m in members:
            parts = PurePosixPath(m.name).parts
            if m.isdir():
                continue
            if not m.isfile():
                raise ValueError(f"not a regular file in archive: {m.name!r}")
            out = dest_parent.joinpath(*parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                raise ValueError(f"cannot read {m.name!r}")
            with src, open(out, "wb") as fh:
                shutil.copyfileobj(src, fh)
    return target


def install(archive: Path, sums_text: str | None, dest_parent: Path) -> Path:
    """Verify `archive` (when `sums_text` is given), extract it and verify the extracted files."""
    archive = Path(archive)
    if sums_text is not None:
        sums = read_checksums(sums_text)
        if archive.name not in sums:
            raise ValueError(f"{archive.name} is not listed in the release {CHECKSUMS}")
        if sha256_file(archive) != sums[archive.name]:
            raise ValueError(f"checksum mismatch for {archive.name}")
    target = safe_extract(archive, dest_parent)
    problems = verify(target)
    if problems:
        raise ValueError("extracted package failed verification: " + "; ".join(problems[:5]))
    return target


def download(owner: str, dest_parent: Path, repo: str = REPO, tag: str | None = None,
             version: str = DATA_VERSION) -> Path:
    """Fetch the release asset and its SHA256SUMS, verify, extract; returns the package directory.

    `version` picks the package (default 1.1.0); `tag` is the release tag, default `v<version>`."""
    asset = asset_name(version)
    tag = tag or f"v{version}"
    dest_parent = Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / asset
        sums = Path(tmp) / CHECKSUMS
        _fetch(release_url(owner, CHECKSUMS, repo, tag), sums)
        _fetch(release_url(owner, asset, repo, tag), archive)
        return install(archive, sums.read_text(encoding="utf-8"), dest_parent)
