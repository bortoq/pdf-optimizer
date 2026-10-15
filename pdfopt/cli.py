"""Command-line interface for the standalone PDF optimizer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .optimize import optimize_pdf


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfopt",
        description="Оптимизация PDF: pypdf, JPEG, MRC и внешние кодировщики.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mode", choices=("lossless", "lossy"),
                      help="режим оптимизации (по умолчанию lossless)")
    mode.add_argument("--lossless", dest="mode", action="store_const",
                      const="lossless", help="оптимизация без потерь")
    parser.add_argument("-q", "--quality", type=int, metavar="1..100",
                        help="минимальный FFmpeg SSIM в процентах; включает lossy")
    dry_run = parser.add_mutually_exclusive_group()
    dry_run.add_argument("--dry-run", dest="dry_run", action="store_true",
                         default=None, help="посчитать результат, не заменяя исходник")
    dry_run.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    recursive = parser.add_mutually_exclusive_group()
    recursive.add_argument("-r", "--recursive", dest="recursive", action="store_true",
                           default=None, help="обрабатывать PDF в подпапках")
    recursive.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--pdf-worker", type=Path,
                        help="путь к hpdf-worker из Hyper Compress")
    parser.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    return parser


def discover(paths: list[Path], recursive: bool) -> list[Path]:
    """Collect unique regular PDF files, skipping symlinks."""
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
            continue
        if target.is_file():
            add(target)
        elif target.is_dir():
            if recursive:
                for directory, dirs, files in os.walk(target, followlinks=False):
                    dirs[:] = [name for name in dirs
                               if not (Path(directory) / name).is_symlink()]
                    for name in sorted(files):
                        add(Path(directory) / name)
            else:
                for child in sorted(target.iterdir()):
                    add(child)
        else:
            raise ValueError(f"путь не найден: {target}")
    return found


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = args.mode or "lossless"
    quality = args.quality if args.quality is not None else 96
    if args.quality is not None:
        if args.mode == "lossless":
            print("pdfopt: -q нельзя сочетать с --lossless", file=sys.stderr)
            return 2
        mode = "lossy"
    if not 1 <= quality <= 100:
        print("pdfopt: quality: ожидается число от 1 до 100", file=sys.stderr)
        return 2
    if args.pdf_worker is not None:
        args.pdf_worker = args.pdf_worker.expanduser().absolute()
    try:
        paths = discover(args.paths, bool(args.recursive))
    except (ValueError, OSError) as exc:
        print(f"pdfopt: {exc}", file=sys.stderr)
        return 2
    if not paths:
        print("Подходящих файлов не найдено.")
        return 0
    dry_run = bool(args.dry_run)
    failures = 0
    total_saved = 0
    for path in paths:
        try:
            status, saved = optimize_pdf(path, mode, dry_run,
                                         args.pdf_worker, quality / 100)
            total_saved += saved
            print(f"{path}: {status}" + (f", {saved} байт" if saved else ""))
        except (OSError, ValueError) as exc:
            failures += 1
            print(f"{path}: ошибка: {exc}", file=sys.stderr)
    print(f"Экономия: {total_saved} байт" +
          (" (оценка, файлы не изменены)" if dry_run else ""))
    return 1 if failures else 0
