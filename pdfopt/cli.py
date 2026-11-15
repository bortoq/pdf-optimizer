"""Command-line interface for the standalone PDF optimizer."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import contextlib
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import signal
import sys
import threading
import time

from . import __version__, cache
from .lossless import ect_available
from .optimize import (
    ALREADY,
    MIN_SSIM,
    MRC_ALLOWANCE,
    Outcome,
    find_worker,
    optimize_pdf,
    remove_work_folders,
)
from .pdfmodel import Skipped


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfopt",
        description="Shrink PDF files. Lossless by default, lossy with -q.",
    )
    parser.add_argument("-q", "--quality", type=int, metavar="1..100",
                        help="lowest SSIM a page may have, in percent; turns on lossy mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="work out the result but do not replace the source file")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="also process PDF files in subfolders")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the result of every file, the size and rejection reason of "
                        "every engine, and warnings")
    parser.add_argument("--decrypt", action="store_true",
                        help="remove the encryption of files that open without a password and "
                        "optimize them; the owner's limits (printing, copying) are lost")
    parser.add_argument("--force", action="store_true",
                        help="process a file even if it is already optimized")
    parser.add_argument("--safe-text", action="store_true",
                        help="with -q, do not use the JBIG2 symbol mode for text: "
                        "files are larger, but similar letters can never be swapped")
    parser.add_argument("--mrc-quality", type=int, metavar="1..100",
                        help="lowest SSIM of MRC text pages, in percent "
                        f"(default: {round(MRC_ALLOWANCE * 100)} below -q)")
    parser.add_argument("-o", "--output", type=Path, metavar="PATH",
                        help="keep the source files: write the result to this file "
                        "(one input file) or to this folder (several input files)")
    parser.add_argument("-j", "--jobs", type=int, metavar="N",
                        help="how many CPU cores to use (default: all)")
    parser.add_argument("--version", action="version", version=f"pdfopt {__version__}")
    parser.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    return parser


WORK_PREFIX = ".pdfopt-"     # work folders and unfinished -o files


def discover(paths: list[Path], recursive: bool,
             rejected: list[str] | None = None) -> list[Path]:
    """Collect unique regular PDF files, skipping symlinks.

    Paths named on the command line that are skipped get a reason in
    ``rejected``.
    """
    seen: set[Path] = set()
    found: list[Path] = []

    def add(path: Path) -> None:
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".pdf":
            return
        absolute = path.absolute()
        if absolute not in seen:
            seen.add(absolute)
            found.append(absolute)

    for target in paths:
        if target.is_symlink():
            if rejected is not None:
                rejected.append(f"{target}: symbolic links are not processed "
                                "(give the path of the file itself)")
            continue
        if target.is_file():
            if target.suffix.lower() != ".pdf" and rejected is not None:
                rejected.append(f"{target}: not a PDF (the name does not end in .pdf)")
            add(target)
        elif target.is_dir():
            if recursive:
                for directory, dirs, files in os.walk(target, followlinks=False):
                    # Not into links, nor into work folders (.pdfopt-*) that a
                    # killed run left behind or another run is using.
                    dirs[:] = sorted(name for name in dirs
                                     if not name.startswith(WORK_PREFIX)
                                     and not (Path(directory) / name).is_symlink())
                    for name in sorted(files):
                        add(Path(directory) / name)
            else:
                for child in sorted(target.iterdir()):
                    add(child)
        else:
            raise ValueError(f"path not found: {target}")
    return found


# Rough peak memory of one file being processed (MB): a lossy run decodes
# 300 dpi pages into arrays, a lossless one mostly streams.
FILE_MEMORY_MB = {"lossy": 700, "lossless": 300}


def _available_mb() -> int | None:
    """Memory that can be used without swapping, or None when unknown."""
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")  # type: ignore[attr-defined]
        size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[attr-defined]
        return pages * size // (2 * 1024 ** 2)
    except (ValueError, OSError, AttributeError):
        return None


def files_at_once(cores: int, files: int, mode: str,
                  available_mb: int | None = None) -> int:
    """Files processed in parallel: bounded by cores, files and memory."""
    count = min(cores, files)
    available = _available_mb() if available_mb is None else available_mb
    if available is not None:
        count = min(count, max(1, available // FILE_MEMORY_MB[mode]))
    return max(1, count)


_PROGRESS = None   # in a worker process: the queue that carries progress to the parent


_KNOWN: set[str] = set()   # keys of the files known to gain nothing (see cache.py)
NO_GAIN = "no change"
NO_GAIN_KNOWN = "no change (known from an earlier run; --force tries again)"


class _Interrupted(BaseException):
    """Ctrl+C or a termination request; unwinds quietly (temporary files are removed)."""


_ENDING = False      # this worker was stopped: it starts no other file
_BUSY = False        # a worker process is inside _process
_STOPPING = False    # ... and is cleaning up after a signal
_AGAIN = False       # ... and was told to stop once more meanwhile


def _stop(signum, frame) -> None:
    """Ctrl+C or a termination request in a worker process."""
    global _STOPPING, _AGAIN
    # Its own helper processes (MRC pages) are told too: a termination
    # request reaches only this process, and the clean-up below would wait
    # for them to finish their pages.
    for child in multiprocessing.active_children():
        with contextlib.suppress(OSError, ValueError):
            child.terminate()
    if _STOPPING:
        _AGAIN = True       # must not break the clean-up; obeyed right after it
    elif not _BUSY:
        os._exit(130)       # between files: nothing to clean up, nobody to answer
    else:
        _STOPPING = True
        raise _Interrupted


_TERMINATED = False


_MAIN_PID = 0           # the process that runs main()
_STOP_SIGNALS = {signal.SIGINT, signal.SIGTERM}


def _terminated(signum, frame) -> None:
    global _TERMINATED
    if os.getpid() != _MAIN_PID:
        # A worker just forked, before its own handler is set: it has
        # nothing to clean up yet, and must not inherit the steps below.
        os._exit(130)
    _TERMINATED = True
    # Once is enough: a second request must not break the clean-up.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt


@contextlib.contextmanager
def _signals_held():
    """Hold Ctrl+C and termination requests while worker processes start.

    A forked worker inherits the mask, so no signal reaches it before it has
    set its own handler (``_init_worker`` releases them). A request that
    comes meanwhile is delivered to this process when the block ends.
    """
    if sys.platform == "win32":     # no signal masks
        yield
        return
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _init_worker(channel, known: set[str]) -> None:
    global _PROGRESS
    _PROGRESS = channel
    _KNOWN.update(known)
    # The terminal sends Ctrl+C to every process; the parent reports it, the
    # workers just stop. Programs started by a worker get the default action.
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    if sys.platform != "win32":
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _STOP_SIGNALS)


def _end_soon(delay: float = 2.0) -> None:
    time.sleep(delay)
    os._exit(130)


def _process(task, emit=None) -> tuple[str, str, int, str, tuple | None, str | None]:
    """Worker: returns (path, outcome, saved, text, quality, key); never raises.

    ``quality`` is (pages, sum of page SSIM, lowest page SSIM) or None.
    ``key`` is set when the file gained nothing: the parent records it, and
    the next run with the same settings does not repeat the work.
    ``emit(path, fraction)`` is told how far the file is; in a worker process
    that goes to the parent through the progress queue.
    """
    (path, mode, dry_run, min_ssim, mrc_ssim, jobs, force, safe_text, verbose,
     decrypt, output, settings) = task
    global _BUSY, _STOPPING, _ENDING
    if _ENDING:
        # Tasks already sit in this worker's queue when the run is stopped
        # (cancelling them is no longer possible): none is started.
        return str(path), "interrupted", 0, "interrupted", None, None
    if emit is None and _PROGRESS is not None:
        def emit(name: str, fraction: float) -> None:
            _PROGRESS.put((name, fraction))

    def log(message: str) -> None:
        if verbose:
            print(f"  {path.name}: {message}", file=sys.stderr, flush=True)

    last = [0.0]

    def progress(fraction: float) -> None:
        if emit is not None and fraction - last[0] >= 0.01:
            last[0] = fraction
            emit(str(path), fraction)

    work = path
    _BUSY = True

    def deliver() -> None:
        """Give the finished copy its name (replaces an older output file)."""
        if work != path:
            os.replace(work, output)

    def discard() -> None:
        # An output file that was there before the run stays as it was.
        if work != path:
            work.unlink(missing_ok=True)

    try:
        key = None
        if not force:
            with contextlib.suppress(OSError):
                key = cache.file_key(path, settings)
        if output is not None and not dry_run:
            # Work on a copy: the source stays as it is. The copy is also the
            # result of files that need no change or are skipped. It gets the
            # name of the output only when it is complete.
            # The name is known before the file exists, so an interruption at
            # any point finds it for the clean-up (one name per process: no two
            # running processes share it).
            output.parent.mkdir(parents=True, exist_ok=True)
            work = output.parent / f"{WORK_PREFIX}{os.getpid()}-{output.name}"
            shutil.copy2(path, work)
        if key is not None and key in _KNOWN:
            outcome = Outcome(NO_GAIN_KNOWN, 0)
        else:
            outcome = optimize_pdf(work, mode, dry_run=dry_run, min_ssim=min_ssim,
                                   jobs=jobs, force=force, safe_text=safe_text,
                                   mrc_ssim=mrc_ssim, decrypt=decrypt, log=log,
                                   progress=progress)
        status, saved = outcome
        scores = getattr(outcome, "scores", ())
        quality = (len(scores), sum(scores), min(scores)) if scores else None
        deliver()
        return str(path), "ok", saved, status, quality, key if status == NO_GAIN else None
    except (KeyboardInterrupt, _Interrupted):
        discard()
        remove_work_folders()
        if _PROGRESS is not None:
            _ENDING = True
            # In a worker process: the run is ending, so this process ends
            # too, shortly after its answer is sent. Waiting for the next
            # task could block for ever on a queue lock held by a worker that
            # was stopped, and a second request may never come (two requests
            # sent at once arrive as one signal).
            threading.Thread(target=_end_soon, daemon=True).start()
        return str(path), "interrupted", 0, "interrupted", None, None
    except Skipped as exc:
        try:
            deliver()
        except OSError as error:
            discard()
            return str(path), "failed", 0, f"error: {error}", None, None
        return str(path), "skipped", 0, f"skipped: {exc}", None, None
    except Exception as exc:    # a damaged file must not stop the other files
        discard()
        text = str(exc) if isinstance(exc, (OSError, ValueError)) else \
            f"unexpected {type(exc).__name__}: {exc}"
        return str(path), "failed", 0, f"error: {text}", None, None
    finally:
        _BUSY = _STOPPING = False
        if _AGAIN:
            # A request to stop came while this one was cleaning up: the pool
            # is being torn down and waits for this process to end. Obeyed
            # however the file ended; waiting for the next task could block
            # for ever on a queue lock that a stopped worker held.
            os._exit(130)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _human(size: int) -> str:
    value = float(size)
    for unit in ("bytes", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} bytes"


def _percent(saved: int, original: int) -> str:
    return f"{100 * saved / original:.1f}%" if original else "0.0%"


class _Bar:
    """One redrawn line on a terminal: overall progress and the saving so far."""

    WIDTH = 24
    EVERY = 0.1   # seconds between redraws

    def __init__(self, total: int):
        self.total = total
        self.shown = False
        self.stamp = 0.0

    def draw(self, fraction: float, done: int, saved: int, original: int, name: str,
             *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.stamp < self.EVERY:
            return
        self.stamp = now
        filled = int(self.WIDTH * fraction)
        columns = shutil.get_terminal_size().columns
        line = (f"[{'#' * filled}{'-' * (self.WIDTH - filled)}] {100 * fraction:3.0f}%  "
                f"{done}/{self.total}  -{_percent(saved, original)} ({_human(saved)})  {name}")
        print("\r" + line[:max(columns - 1, 20)], end="\033[K", file=sys.stderr, flush=True)
        self.shown = True

    def clear(self) -> None:
        if self.shown:
            print("\r", end="\033[K", file=sys.stderr, flush=True)
            self.shown = False


def _main_thread() -> bool:
    import threading
    return threading.current_thread() is threading.main_thread()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name, value in (("quality", args.quality), ("mrc-quality", args.mrc_quality)):
        if value is not None and not 1 <= value <= 100:
            print(f"pdfopt: {name}: expected a number from 1 to 100", file=sys.stderr)
            return 2
    mode = "lossy" if args.quality is not None else "lossless"
    min_ssim = args.quality / 100 if args.quality is not None else MIN_SSIM
    mrc_ssim = args.mrc_quality / 100 if args.mrc_quality is not None else None
    if args.jobs is not None and args.jobs < 1:
        print("pdfopt: jobs: expected a number of at least 1", file=sys.stderr)
        return 2
    rejected: list[str] = []
    try:
        paths = discover(args.paths, args.recursive, rejected)
    except (ValueError, OSError) as exc:
        print(f"pdfopt: {exc}", file=sys.stderr)
        return 2
    for line in rejected:
        print(f"pdfopt: skipped {line}", file=sys.stderr)
    if not paths:
        print("No PDF files found.")
        return 2 if rejected else 0
    targets: dict[Path, Path | None] = dict.fromkeys(paths)
    if args.output is not None:
        if len(paths) == 1 and not args.output.is_dir():
            targets[paths[0]] = args.output.absolute()
        else:
            names = [path.name for path in paths]
            if len(set(names)) != len(names):
                print("pdfopt: -o: some input files have the same name", file=sys.stderr)
                return 2
            if args.output.is_file():
                print("pdfopt: -o: several input files need a folder", file=sys.stderr)
                return 2
            for path in paths:
                targets[path] = (args.output / path.name).absolute()
        if any(target == path for path, target in targets.items()):
            print("pdfopt: -o: the output is the same as the input file", file=sys.stderr)
            return 2
    if args.verbose and not ect_available():
        print("pdfopt: warning: the ECT module is not built "
              "(python3 setup.py build_ext --inplace), streams are packed with zlib",
              file=sys.stderr)
    if args.verbose and find_worker() is None:
        print("pdfopt: warning: hpdf-worker not found in PATH, "
              "Hyper Compress is not used", file=sys.stderr)

    cores = args.jobs or os.cpu_count() or 1
    parallel = files_at_once(cores, len(paths), mode)
    per_file = max(1, cores // parallel)
    settings = cache.settings_key(mode, min_ssim, mrc_ssim, args.safe_text, args.decrypt)
    no_gain = set() if args.force else cache.load()
    _KNOWN.clear()
    _KNOWN.update(no_gain)
    gained_nothing: list[str] = []
    tasks = [(path, mode, args.dry_run, min_ssim, mrc_ssim, per_file, args.force,
              args.safe_text, args.verbose, args.decrypt, targets[path], settings)
             for path in paths]
    sizes = {str(path): _size(path) for path in paths}
    weight = sum(sizes.values()) or 1
    failures = skipped = known = total_saved = done_size = done = 0
    pages = ssim_sum = 0
    lowest = 1.0
    fractions: dict[str, float] = dict.fromkeys(sizes, 0.0)
    bar = None if args.verbose or not sys.stderr.isatty() else _Bar(len(paths))
    current = ""
    interrupted: list[str] = []

    def overall() -> float:
        return sum(sizes[name] * value for name, value in fractions.items()) / weight

    def progress(name: str, fraction: float) -> None:
        nonlocal current
        fractions[name] = max(fractions[name], min(fraction, 0.999))
        current = Path(name).name
        if bar:
            bar.draw(overall(), done, total_saved, done_size, current)

    def report(result) -> None:
        nonlocal failures, skipped, known, total_saved, done_size, done
        nonlocal pages, ssim_sum, lowest
        name, outcome, saved, text, quality, key = result
        if key is not None:
            gained_nothing.append(key)
        if outcome == "interrupted":
            interrupted.append(name)
            return
        done += 1
        done_size += sizes[name]
        total_saved += saved
        fractions[name] = 1.0
        if quality:
            pages += quality[0]
            ssim_sum += quality[1]
            lowest = min(lowest, quality[2])
        if bar:
            bar.clear()
        if outcome == "failed":
            failures += 1
            print(f"{name}: {text}", file=sys.stderr)
        else:
            skipped += outcome == "skipped"
            known += text == ALREADY
            if args.verbose or outcome == "skipped":
                print(f"{name}: {text}" + (f", {saved} bytes" if saved else ""))
        if bar:
            bar.draw(overall(), done, total_saved, done_size, Path(name).name, force=True)

    if bar:
        bar.draw(0.0, 0, 0, 0, "", force=True)
    # A termination request (kill, docker stop, timeout) ends the run like
    # Ctrl+C: work folders are removed and the sources stay unchanged.
    global _TERMINATED, _MAIN_PID
    _TERMINATED = False
    _MAIN_PID = os.getpid()
    previous = signal.signal(signal.SIGTERM, _terminated) if _main_thread() else None
    try:
        if parallel > 1:
            channel: multiprocessing.Queue = multiprocessing.Queue()
            pool = ProcessPoolExecutor(max_workers=parallel, initializer=_init_worker,
                                       initargs=(channel, no_gain))
            running: set = set()
            # The interruption may come at any point here, also while the
            # tasks are handed out: the workers are told in every case, or
            # the pool would wait for them to finish their files.
            try:
                with _signals_held():
                    for task in tasks:
                        running.add(pool.submit(_process, task))
                while running:
                    finished, running = wait(running, timeout=0.1,
                                             return_when=FIRST_COMPLETED)
                    try:
                        while True:
                            progress(*channel.get_nowait())
                    except queue.Empty:
                        pass
                    for future in finished:
                        report(future.result())
                    if interrupted:     # a worker was stopped: stop the rest
                        raise KeyboardInterrupt
            except KeyboardInterrupt:
                for future in running:
                    future.cancel()
                if _TERMINATED:
                    # Unlike Ctrl+C, a termination request reaches only this
                    # process: pass it on.
                    for worker in list((getattr(pool, "_processes", None) or {}).values()):
                        with contextlib.suppress(OSError, ValueError):
                            worker.terminate()
                # Workers were stopped by the same Ctrl+C; collect them. One
                # that was stopped before it could answer has no result.
                for future in wait(running)[0]:
                    if not future.cancelled():
                        with contextlib.suppress(Exception):
                            report(future.result())
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        else:
            for task in tasks:
                report(_process(task, progress))
                if interrupted:
                    raise KeyboardInterrupt
    except (KeyboardInterrupt, BrokenProcessPool) as stopped:
        if bar:
            bar.clear()
        if isinstance(stopped, BrokenProcessPool) and not _TERMINATED:
            # Killed from outside, most often by the system when memory ran out.
            print(f"pdfopt: a worker process died, {done} of {len(paths)} files "
                  "finished; the others are unchanged (try fewer cores with -j)",
                  file=sys.stderr)
            return 1
        print(f"pdfopt: interrupted, {done} of {len(paths)} files "
              "finished; the others are unchanged", file=sys.stderr)
        return 130
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        cache.add(gained_nothing)
    if bar:
        bar.clear()
    print(f"Saved: {_human(total_saved)} ({_percent(total_saved, done_size)})" +
          (f" SSIM {ssim_sum / pages:.4f} (lowest page {lowest:.4f})" if pages else "") +
          (" (estimate, files not changed)" if args.dry_run else "") +
          (f"; skipped: {skipped}" if skipped else "") +
          (f"; already optimized: {known} (use --force to process again)" if known else "") +
          (f"; errors: {failures}" if failures else ""))
    return 1 if failures else 0


def run() -> None:
    """The ``pdfopt`` command: ``main`` with the process exit code.

    After an interruption the work is unwound and every temporary file is
    gone; the process then ends at once instead of finalizing the interpreter,
    which can crash in native code that was cut off in the middle (seen as a
    segmentation fault on exit under heavy load).
    """
    code = main()
    if code == 130:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        os._exit(130)
    sys.exit(code)
