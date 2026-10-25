"""Page rendering (poppler) and visual comparison gates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .ssim import ssim
from .tools import ToolError, last_line, require, run

SSIM_DPI = 200
EXACT_DPI = 100
CHUNK = 8


def render(pdf: Path, first: int, last: int, directory: Path, prefix: str,
           *, dpi: int, png: bool = False) -> list[Path]:
    """Render pages ``first..last`` (1-based, inclusive) to image files."""
    directory.mkdir(parents=True, exist_ok=True)
    done = run([require("pdftoppm"), "-f", str(first), "-l", str(last),
                "-r", str(dpi), "-cropbox", *(["-png"] if png else []),
                str(pdf), str(directory / prefix)])
    suffix = ".png" if png else ".ppm"
    files = sorted(directory.glob(prefix + "-*" + suffix))
    if done.returncode != 0 or len(files) != last - first + 1:
        raise ToolError("pdftoppm: " + last_line(done))
    return files


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _runs(pages: list[int]) -> list[list[int]]:
    """Group sorted page numbers into consecutive runs of at most CHUNK."""
    runs: list[list[int]] = []
    for page in sorted(pages):
        if runs and page == runs[-1][-1] + 1 and len(runs[-1]) < CHUNK:
            runs[-1].append(page)
        else:
            runs.append([page])
    return runs


class VisualGate:
    """Compares rendered pages of candidates against the source document.

    The reference renders are produced once (PNG, in parallel) and reused for
    every candidate; candidates are rendered in small chunks and compared
    immediately, so a failing page stops the check early.
    """

    def __init__(self, source: Path, scratch: Path, minimum: float, jobs: int = 1):
        self.source = source
        self.dir = scratch / "gate"
        self.minimum = minimum
        self.jobs = max(1, jobs)
        self.reference: dict[int, Path] = {}
        self._counter = 0

    def prepare(self, pages: list[int]) -> None:
        """Render the reference for 1-based ``pages`` that are still missing."""
        todo = [p for p in pages if p not in self.reference]
        if not todo:
            return

        def work(run_pages: list[int]) -> None:
            files = render(self.source, run_pages[0], run_pages[-1],
                           self.dir / "ref", f"r{run_pages[0]}",
                           dpi=SSIM_DPI, png=True)
            for page, file in zip(run_pages, files):
                self.reference[page] = file

        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            list(pool.map(work, _runs(todo)))

    def failures(self, candidate: Path, pages: list[int], *, stop_early: bool = False,
                 minimum: float | None = None) -> list[int]:
        """1-based pages whose SSIM against the source is below the minimum
        (the gate's own unless ``minimum`` is given)."""
        minimum = self.minimum if minimum is None else minimum
        self.prepare(pages)
        self._counter += 1
        bad: list[int] = []
        base = self.dir / f"c{self._counter}"

        def work(run_pages: list[int]) -> list[int]:
            if stop_early and bad:
                return []
            files = render(candidate, run_pages[0], run_pages[-1], base,
                           f"p{run_pages[0]}", dpi=SSIM_DPI)
            failed = []
            for page, file in zip(run_pages, files):
                try:
                    result = _load(file)
                    original = _load(self.reference[page])
                    ok = (result.shape == original.shape and
                          ssim(original, result) >= minimum)
                except (OSError, ValueError):
                    ok = False
                if not ok:
                    failed.append(page)
                file.unlink(missing_ok=True)
            return failed

        try:
            with ThreadPoolExecutor(max_workers=self.jobs) as pool:
                for failed in pool.map(work, _runs(pages)):
                    bad.extend(failed)
        finally:
            shutil.rmtree(base, ignore_errors=True)
        return sorted(bad)

    def score(self, candidate: Path, page: int) -> float:
        self.prepare([page])
        files = render(candidate, page, page, self.dir / "score", "s", dpi=SSIM_DPI)
        try:
            return ssim(_load(self.reference[page]), _load(files[0]))
        finally:
            shutil.rmtree(self.dir / "score", ignore_errors=True)


def identical_pages(first: Path, second: Path, pages: list[int], scratch: Path,
                    jobs: int = 1) -> bool:
    """True when the listed 1-based pages render pixel-for-pixel the same."""
    base = scratch / "exact"

    def work(run_pages: list[int]) -> bool:
        a = render(first, run_pages[0], run_pages[-1], base / "a",
                   f"a{run_pages[0]}", dpi=EXACT_DPI)
        b = render(second, run_pages[0], run_pages[-1], base / "b",
                   f"b{run_pages[0]}", dpi=EXACT_DPI)
        same = all(np.array_equal(_load(x), _load(y)) for x, y in zip(a, b))
        for file in (*a, *b):
            file.unlink(missing_ok=True)
        return same

    try:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            return all(pool.map(work, _runs(pages)))
    finally:
        shutil.rmtree(base, ignore_errors=True)
