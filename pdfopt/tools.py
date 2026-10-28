"""External tool discovery and a single subprocess wrapper."""

from __future__ import annotations

from functools import cache
from pathlib import Path
import shutil
import subprocess

TIMEOUT = 600


class ToolError(ValueError):
    """An external program failed or is missing."""


@cache
def which(name: str) -> str | None:
    return shutil.which(name)


def require(name: str) -> str:
    found = which(name)
    if found is None:
        raise ToolError(f"{name} not found")
    return found


def run(args: list[str], *, timeout: float = TIMEOUT,
        stdout_to: Path | None = None) -> subprocess.CompletedProcess:
    """Run a tool without a shell; never inherits stdin.

    With ``stdout_to`` the standard output is written to that file in binary
    mode (used by encoders that only write to stdout).
    """
    try:
        if stdout_to is not None:
            with stdout_to.open("wb") as sink:
                return subprocess.run(args, stdout=sink,
                                      stderr=subprocess.PIPE, check=False,
                                      timeout=timeout,
                                      stdin=subprocess.DEVNULL)
        return subprocess.run(args, capture_output=True, text=True,
                              check=False, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolError(f"{Path(args[0]).name}: {exc}") from exc


def last_line(completed: subprocess.CompletedProcess) -> str:
    """The most useful diagnostic line of a finished process."""
    text = completed.stderr or completed.stdout or ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    lines = text.strip().splitlines()
    return lines[-1] if lines else f"exit code {completed.returncode}"
