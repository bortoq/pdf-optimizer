"""Choose the smallest verified PDF and replace the source atomically."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass, field
import itertools
import os
from pathlib import Path
import shutil
import tempfile
import time

import pikepdf

from . import jpegopt, lossless, mrc
from .pdfmodel import (
    MARKER_VERSION,
    Fingerprint,
    Skipped,
    compare,
    has_signature,
    open_pdf,
    read_marker,
    set_marker,
)
from .render import VisualGate, identical_pages
from .tools import ToolError, last_line, require, run, which

MIN_SSIM = 0.96
# Default allowance for MRC text pages below -q: a bilevel stencil cannot
# reproduce the soft edges of scanned glyphs (a 300 dpi scan tops out near
# SSIM 0.95 even over a perfect background), just like DjVu's JB2.
MRC_ALLOWANCE = 0.06
TIMEOUT = 600
HYPER_TOKENS = {
    "lossless": ("3=0", "10=1", "13=1", "14=1", "16=1", "17=1"),
    "lossy": (
        "1=150", "2=50", "3=4", "8=1", "18=1", "7=1", "63=10",
        "6=1", "10=1", "13=1", "14=1", "16=1", "17=1", "40=1",
    ),
}
# Ghostscript's QFactor grows as JPEG compression becomes stronger.
GHOSTSCRIPT_PROFILES = ((300, 0.35), (240, 0.50), (200, 0.65))

Log = Callable[[str], None]


@dataclass
class Candidate:
    name: str
    path: Path
    size: int
    # How the candidate is verified after the content/metadata comparison:
    exact_pages: list[int] = field(default_factory=list)   # pixel-identical
    gate_pages: list[int] | None = field(default_factory=list)  # SSIM; None=all
    content: bool = False        # page content streams must be byte-identical
    metadata: bool = False
    verified: bool = False


@dataclass
class Task:
    """One engine run: produces candidates or raises."""

    label: str
    run: Callable[[], list[Candidate]]


# ------------------------------------------------------------------ worker --

def find_worker() -> Path | None:
    """``hpdf-worker`` from PATH, or None (the caller warns)."""
    found = which("hpdf-worker")
    return Path(found) if found else None


def _make_hyper(worker: Path, source: Path, target: Path, mode: str) -> None:
    done = run([str(worker), str(source), str(target), *HYPER_TOKENS[mode]])
    if done.returncode != 0 or not target.is_file():
        raise ToolError("Hyper Compress: " + last_line(done))
    if "SIGNED_SKIP" in (done.stderr or ""):
        raise ToolError("PDF: Hyper Compress обнаружил электронную подпись")


def _make_ghostscript(source: Path, target: Path, dpi: int, qfactor: float) -> None:
    done = run([
        require("gs"), "-dSAFER", "-dBATCH", "-dNOPAUSE", "-q",
        "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.6",
        "-dPDFSETTINGS=/prepress", "-dAutoRotatePages=/None",
        "-dDetectDuplicateImages=true", "-dPassThroughJPEGImages=false",
        "-dPassThroughJPXImages=false",
        "-dAutoFilterColorImages=false", "-dAutoFilterGrayImages=false",
        "-dColorImageFilter=/DCTEncode", "-dGrayImageFilter=/DCTEncode",
        "-dDownsampleColorImages=true", "-dDownsampleGrayImages=true",
        "-dDownsampleMonoImages=false",
        "-dColorImageDownsampleType=/Bicubic",
        "-dGrayImageDownsampleType=/Bicubic",
        f"-dColorImageResolution={dpi}", f"-dGrayImageResolution={dpi}",
        "-sOutputFile=" + str(target), "-c",
        f"<< /ColorImageDict << /QFactor {qfactor} >> "
        f"/GrayImageDict << /QFactor {qfactor} >> >> setdistillerparams",
        "-f", str(source)])
    if done.returncode != 0 or not target.is_file():
        raise ToolError("Ghostscript: " + last_line(done))


# ----------------------------------------------------------- orchestration --

class _Run:
    """State of one optimize_pdf call."""

    def __init__(self, path: Path, mode: str, min_ssim: float, jobs: int,
                 scratch: Path, baseline: Fingerprint, log: Log, safe_text: bool = False,
                 mrc_ssim: float | None = None):
        self.path = path
        self.mode = mode
        self.min_ssim = min_ssim
        self.mrc_ssim = mrc_ssim
        self.jobs = jobs
        self.scratch = scratch
        self.baseline = baseline
        self.log = log
        self.safe_text = safe_text
        self.size = path.stat().st_size
        self.errors: list[str] = []
        self.gate: VisualGate | None = None
        self._sub = itertools.count(1)   # tasks run in threads

    def directory(self, name: str) -> Path:
        path = self.scratch / f"{name}-{next(self._sub)}"
        path.mkdir()
        return path

    def visual_gate(self) -> VisualGate:
        if self.gate is None:
            self.gate = VisualGate(self.path, self.scratch, self.min_ssim, self.jobs)
        return self.gate

    def fail(self, name: str, reason) -> None:
        self.errors.append(f"{name}: {reason}")
        self.log(f"{name}: отклонён — {reason}")

    # -- the lossless stage ------------------------------------------------

    def lossless_stage(self) -> Candidate | None:
        """Build and verify the pure lossless result (also the lossy base)."""
        for label, options in (("lossless", {}),
                               ("lossless-safe", {"transforms": False})):
            work = self.directory(label)
            target = work / "out.pdf"
            try:
                report = lossless.build(self.path, target, work, self.jobs, **options)
                candidate = Candidate(label, target, target.stat().st_size,
                                      exact_pages=[p + 1 for p in sorted(report.visual_pages)],
                                      content=True, metadata=True)
                self.verify(candidate)
                candidate.verified = True
                self.log(f"{label}: {candidate.size} байт ({', '.join(report.notes) or '—'})")
                return candidate
            except Exception as exc:
                self.fail(label, exc)
        return None

    # -- verification ------------------------------------------------------

    def verify(self, candidate: Candidate) -> None:
        with open_pdf(candidate.path) as pdf:
            fingerprint = Fingerprint.read(pdf, candidate.path)
        compare(fingerprint, self.baseline, content=candidate.content,
                metadata=candidate.metadata)
        if candidate.exact_pages and not identical_pages(
                self.path, candidate.path, candidate.exact_pages,
                self.directory("exact"), self.jobs):
            raise ValueError("PDF: отрисовка изменилась в режиме без потерь")
        if candidate.gate_pages is None or candidate.gate_pages:
            pages = (list(range(1, len(self.baseline.pages) + 1))
                     if candidate.gate_pages is None else candidate.gate_pages)
            bad = self.visual_gate().failures(candidate.path, pages, stop_early=True)
            if bad:
                raise ValueError(f"страница {bad[0]} ниже SSIM {self.min_ssim:.2f}")

    # -- lossy engines -----------------------------------------------------

    def jpeg_stage(self, pdf: pikepdf.Pdf, target: Path, work: Path) -> bool:
        """Re-encode images of ``pdf``, verify, undo failing pages, save.

        Returns False when nothing was left to save.
        """
        counts, changes = jpegopt.optimize_jpegs(pdf, work, self.min_ssim, self.jobs)
        if not changes:
            return False
        self.log("JPEG: " + ", ".join(f"{name}×{count}" for name, count in counts.items()))
        lossless.save(pdf, target, deflated=True)
        pages = sorted({page + 1 for change in changes for page in change.pages})
        bad = set(self.visual_gate().failures(target, pages))
        if bad:
            kept = []
            for change in changes:
                if any(page + 1 in bad for page in change.pages):
                    jpegopt.revert(change)
                else:
                    kept.append(change)
            self.log(f"JPEG: возвращены картинки страниц {sorted(bad)[:8]}")
            if not kept:
                return False
            lossless.save(pdf, target, deflated=True)
        return True

    def mrc_engine(self, stage: Path) -> list[Candidate]:
        out: list[Candidate] = []
        work = self.directory("mrc")
        plain = work / "mrc.pdf"
        chained = work / "mrc-jpeg.pdf"
        saved = []
        layered = []        # set once the MRC result is verified and saved

        def chain(pdf: pikepdf.Pdf) -> None:
            layered.append(True)
            saved.append(self.jpeg_stage(pdf, chained, work))

        try:
            mrc.build(stage, plain, work, self.min_ssim, self.jobs,
                      self.visual_gate(), self.log, then=chain,
                      symbols=not self.safe_text, relaxed=self.mrc_ssim)
        except Exception as exc:   # any failure: keep what is already verified
            self.log(f"{'JPEG' if layered else 'MRC'}: {exc}")
            saved.clear()
            if not layered:
                # Without layered pages the JPEG stage may still help.
                try:
                    with open_pdf(stage) as pdf:
                        saved.append(self.jpeg_stage(pdf, chained, work))
                except Exception as error:
                    self.fail("JPEG", error)
        if layered:
            out.append(Candidate("MRC", plain, plain.stat().st_size))
        if saved and saved[-1]:
            name = "MRC+JPEG" if out else "JPEG"
            # The pages of the JPEG stage were gated inside jpeg_stage.
            out.append(Candidate(f"{name} SSIM≥{self.min_ssim:.2f}", chained,
                                 chained.stat().st_size))
        return out

    def external_tasks(self, worker: Path | None) -> list[Task]:
        tasks: list[Task] = []
        pages = list(range(1, len(self.baseline.pages) + 1))
        if worker is not None:
            lossy = self.mode == "lossy"

            def hyper() -> list[Candidate]:
                target = self.directory("hyper") / "out.pdf"
                _make_hyper(worker, self.path, target, self.mode)
                # In lossless mode the worker's output must render pixel-for-pixel.
                return [Candidate("Hyper Compress", target, target.stat().st_size,
                                  gate_pages=None if lossy else [],
                                  exact_pages=[] if lossy else pages,
                                  metadata=not lossy)]

            tasks.append(Task("Hyper Compress", hyper))
        if (self.mode == "lossy" and not self.baseline.interactive and
                not self.baseline.has_text_layer and which("gs")):
            for dpi, qfactor in GHOSTSCRIPT_PROFILES:
                def ghostscript(dpi=dpi, qfactor=qfactor) -> list[Candidate]:
                    target = self.directory("gs") / "out.pdf"
                    _make_ghostscript(self.path, target, dpi, qfactor)
                    return [Candidate(f"Ghostscript {dpi} dpi/Q{qfactor}", target,
                                      target.stat().st_size, gate_pages=None)]

                tasks.append(Task(f"Ghostscript {dpi}", ghostscript))
        return tasks


def _pick(run_: _Run, candidates: list[Candidate]) -> Candidate | None:
    """Smallest candidate that is smaller than the source and verifies."""
    for candidate in sorted(candidates, key=lambda c: c.size):
        if candidate.size >= run_.size:
            continue
        try:
            if not candidate.verified:
                run_.verify(candidate)
        except Exception as exc:
            run_.fail(candidate.name, exc)
            continue
        return candidate
    return None


def _stamp(candidate: Candidate, lossy: bool, scratch: Path) -> Path:
    """Write the optimizer marker into the winner (a cheap lossless re-save)."""
    target = scratch / "stamped.pdf"
    with open_pdf(candidate.path) as pdf:
        set_marker(pdf, lossy)
        lossless.save(pdf, target, deflated=True)
    return target


def _timed(log: Log | None) -> Log:
    """Prefix log lines with the elapsed time."""
    start = time.monotonic()
    if log is None:
        return lambda message: None
    return lambda message: log(f"{time.monotonic() - start:6.1f} с  {message}")


def optimize_pdf(path: Path, mode: str = "lossless", *, dry_run: bool = False,
                 min_ssim: float = MIN_SSIM, jobs: int | None = None,
                 force: bool = False, safe_text: bool = False,
                 mrc_ssim: float | None = None,
                 log: Log | None = None) -> tuple[str, int]:
    """Choose the smallest verified PDF and replace the source atomically.

    Raises ``Skipped`` for files that are deliberately left alone (signed or
    encrypted) and ``ValueError`` for files that could not be processed.
    ``safe_text`` codes MRC ink masks losslessly (no JBIG2 symbol
    dictionaries). ``mrc_ssim`` is the SSIM required of layered text pages
    (default ``min_ssim - MRC_ALLOWANCE``; none below a perfect 1.0). All
    cores are used unless ``jobs`` says otherwise.
    """
    if mrc_ssim is None:
        mrc_ssim = min_ssim - MRC_ALLOWANCE if min_ssim < 1.0 else min_ssim
    if mode not in ("lossless", "lossy"):
        raise ValueError(f"PDF: неизвестный режим: {mode}")
    jobs = max(1, jobs or os.cpu_count() or 1)
    log = _timed(log)
    source_stat = path.stat()
    with open_pdf(path) as pdf:
        if has_signature(pdf):
            raise Skipped("документ с электронной подписью не изменён")
        marker = read_marker(pdf)
        baseline = Fingerprint.read(pdf, path)
    lossy = mode == "lossy"
    if not force and marker and marker[0] >= MARKER_VERSION and (marker[1] or not lossy):
        return "уже оптимизирован", 0
    worker = find_worker()
    if lossy and which("pdftoppm") is None:
        raise ValueError("PDF: для проверки результата нужен pdftoppm")
    if worker is not None and which("pdftoppm") is None:
        raise ValueError("PDF: для проверки Hyper Compress нужен pdftoppm")

    with tempfile.TemporaryDirectory(prefix=".pdfopt-", dir=path.parent) as directory:
        scratch = Path(directory)
        state = _Run(path, mode, min_ssim, jobs, scratch, baseline, log, safe_text,
                     mrc_ssim)
        candidates: list[Candidate] = []
        stage = state.lossless_stage()
        if stage is not None:
            candidates.append(stage)
        tasks: list[Task] = []
        if lossy:
            base = stage.path if stage is not None else path
            tasks.append(Task("MRC", lambda: state.mrc_engine(base)))
        tasks.extend(state.external_tasks(worker))

        def run_task(task: Task) -> list[Candidate]:
            try:
                return task.run()
            except Exception as exc:
                state.fail(task.label, exc)
                return []

        if jobs >= 4 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(3, len(tasks))) as pool:
                for found in pool.map(run_task, tasks):
                    candidates.extend(found)
        else:
            for task in tasks:
                candidates.extend(run_task(task))

        best = _pick(state, candidates)
        if best is None:
            if not candidates and state.errors:
                raise ValueError("PDF: " + "; ".join(state.errors))
            return "без изменений", 0
        final = _stamp(best, lossy, scratch)
        size = final.stat().st_size
        if size >= source_stat.st_size:
            return "без изменений", 0
        saving = source_stat.st_size - size
        if dry_run:
            return f"возможная экономия ({best.name})", saving
        current = path.stat()
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
                source_stat.st_dev, source_stat.st_ino, source_stat.st_size,
                source_stat.st_mtime_ns):
            raise ValueError("исходный файл изменился во время обработки")
        shutil.copystat(path, final)   # mode, times and extended attributes
        if hasattr(os, "chown") and (current.st_uid, current.st_gid) != (
                os.getuid(), os.getgid()):
            # Only root can do this; anyone else only replaces their own files.
            with contextlib.suppress(PermissionError):
                os.chown(final, current.st_uid, current.st_gid)
        os.replace(final, path)
        return f"сжато ({best.name})", saving
