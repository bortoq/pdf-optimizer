"""Command-line interface for the standalone PDF optimizer."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import shutil
import sys

from . import __version__
from .lossless import ect_available
from .optimize import ALREADY, MIN_SSIM, MRC_ALLOWANCE, find_worker, optimize_pdf
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
                    dirs[:] = sorted(name for name in dirs
                                     if not (Path(directory) / name).is_symlink())
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
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (2 * 1024 ** 2)
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


def _process(task) -> tuple[str, str, int, str]:
    """Worker: returns (path, outcome, saved, text); never raises."""
    (path, mode, dry_run, min_ssim, mrc_ssim, jobs, force, safe_text, verbose,
     decrypt, output) = task

    def log(message: str) -> None:
        if verbose:
            print(f"  {path.name}: {message}", file=sys.stderr, flush=True)

    work = path
    try:
        if output is not None and not dry_run:
            # Work on a copy: the source stays as it is. The copy is also the
            # result of files that need no change or are skipped.
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            work = output
        status, saved = optimize_pdf(work, mode, dry_run=dry_run, min_ssim=min_ssim,
                                     jobs=jobs, force=force, safe_text=safe_text,
                                     mrc_ssim=mrc_ssim, decrypt=decrypt, log=log)
        return str(path), "ok", saved, status
    except Skipped as exc:
        return str(path), "skipped", 0, f"skipped: {exc}"
    except (OSError, ValueError) as exc:
        if work != path:
            work.unlink(missing_ok=True)
        return str(path), "failed", 0, f"error: {exc}"


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
    """One redrawn line on a terminal: files done and the saving so far."""

    WIDTH = 24

    def __init__(self, total: int):
        self.total = total
        self.shown = False
        self.draw(0, 0, 0, "")

    def draw(self, done: int, saved: int, original: int, name: str) -> None:
        filled = self.WIDTH * done // self.total
        columns = shutil.get_terminal_size().columns
        line = (f"[{'#' * filled}{'-' * (self.WIDTH - filled)}] {done}/{self.total}  "
                f"-{_percent(saved, original)} ({_human(saved)})  {name}")
        print("\r" + line[:max(columns - 1, 20)], end="\033[K", file=sys.stderr, flush=True)
        self.shown = True

    def clear(self) -> None:
        if self.shown:
            print("\r", end="\033[K", file=sys.stderr, flush=True)
            self.shown = False


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
    tasks = [(path, mode, args.dry_run, min_ssim, mrc_ssim, per_file, args.force,
              args.safe_text, args.verbose, args.decrypt, targets[path]) for path in paths]
    sizes = {str(path): _size(path) for path in paths}
    failures = skipped = known = total_saved = done_size = done = 0
    bar = None if args.verbose or not sys.stderr.isatty() else _Bar(len(paths))

    def report(result) -> None:
        nonlocal failures, skipped, known, total_saved, done_size, done
        name, outcome, saved, text = result
        done += 1
        done_size += sizes[name]
        total_saved += saved
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
            bar.draw(done, total_saved, done_size, Path(name).name)

    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            for result in pool.map(_process, tasks):
                report(result)
    else:
        for task in tasks:
            report(_process(task))
    if bar:
        bar.clear()
    print(f"Saved: {_human(total_saved)} ({_percent(total_saved, done_size)})" +
          (" (estimate, files not changed)" if args.dry_run else "") +
          (f"; skipped: {skipped}" if skipped else "") +
          (f"; already optimized: {known} (use --force to process again)" if known else "") +
          (f"; errors: {failures}" if failures else ""))
    return 1 if failures else 0
