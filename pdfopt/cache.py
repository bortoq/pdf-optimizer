"""Remembers files that gained nothing, so a later run does not repeat the work.

A file that cannot be made smaller gets no marker (it is not rewritten), and
without this every run over a library would analyse it again. The record is
keyed by the content of the file and by everything the result depends on: the
pdfopt version, the mode and thresholds, and the external programs found.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from . import __version__
from .tools import which

NAME = "no-gain.json"
LIMIT = 50_000              # records kept; the oldest go first
_TOOLS = ("pdftoppm", "jbig2", "opj_compress", "cjpeg", "gs", "hpdf-worker")


def folder() -> Path:
    """The cache folder of the user (not created here)."""
    base = os.environ.get("XDG_CACHE_HOME")
    if not base and sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / ".cache") / "pdfopt"


def settings_key(mode: str, min_ssim: float, mrc_ssim: float | None,
                 safe_text: bool, decrypt: bool) -> str:
    """Everything besides the file that decides whether a run can gain."""
    from .lossless import ect_available
    tools = ",".join(name for name in _TOOLS if which(name))
    return (f"{__version__}|{mode}|{min_ssim:.4f}|{mrc_ssim}|{safe_text}|{decrypt}|"
            f"ect={ect_available()}|{tools}")


def file_key(path: Path, settings: str) -> str:
    digest = hashlib.sha256(settings.encode("utf-8"))
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load() -> set[str]:
    """Keys of the files known to gain nothing (empty when there is no cache)."""
    try:
        data = json.loads((folder() / NAME).read_text(encoding="ascii"))
        return {key for key in data if isinstance(key, str)}
    except (OSError, ValueError, TypeError):
        return set()


def add(keys: list[str]) -> None:
    """Record more files; failures are ignored (the cache is only a shortcut)."""
    if not keys:
        return
    target = folder() / NAME
    try:
        try:
            known = [key for key in json.loads(target.read_text(encoding="ascii"))
                     if isinstance(key, str)]
        except (OSError, ValueError, TypeError):
            known = []
        seen = set(known)
        known += [key for key in dict.fromkeys(keys) if key not in seen]
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".no-gain-", dir=target.parent)
        try:
            with os.fdopen(handle, "w", encoding="ascii") as stream:
                json.dump(known[-LIMIT:], stream)
            os.replace(name, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(name)
            raise
    except OSError:
        pass
