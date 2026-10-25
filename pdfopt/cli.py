"""Command-line interface for the standalone PDF optimizer."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys

from .lossless import ect_available
from .optimize import MIN_SSIM, MRC_ALLOWANCE, find_worker, optimize_pdf
from .pdfmodel import Skipped


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfopt",
        description="Оптимизация PDF: без потерь по умолчанию, с потерями при -q.",
    )
    parser.add_argument("-q", "--quality", type=int, metavar="1..100",
                        help="минимальный SSIM страницы в процентах; включает lossy")
    parser.add_argument("--dry-run", action="store_true",
                        help="посчитать результат, не заменяя исходник")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="обрабатывать PDF в подпапках")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="печатать размеры и причины отказа по каждому движку")
    parser.add_argument("--force", action="store_true",
                        help="обработать файл, даже если он уже оптимизирован")
    parser.add_argument("--safe-text", action="store_true",
                        help="при -q кодировать маску текста без символьного режима "
                        "JBIG2: крупнее, зато без подмены похожих букв")
    parser.add_argument("--mrc-quality", type=int, metavar="1..100",
                        help="минимальный SSIM текстовых страниц MRC в процентах "
                        f"(по умолчанию на {round(MRC_ALLOWANCE * 100)} ниже -q)")
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
                    dirs[:] = sorted(name for name in dirs
                                     if not (Path(directory) / name).is_symlink())
                    for name in sorted(files):
                        add(Path(directory) / name)
            else:
                for child in sorted(target.iterdir()):
                    add(child)
        else:
            raise ValueError(f"путь не найден: {target}")
    return found


def _process(task) -> tuple[str, str, int, str]:
    """Worker: returns (path, outcome, saved, text); never raises."""
    path, mode, dry_run, min_ssim, mrc_ssim, jobs, force, safe_text, verbose = task

    def log(message: str) -> None:
        if verbose:
            print(f"  {path.name}: {message}", file=sys.stderr, flush=True)

    try:
        status, saved = optimize_pdf(path, mode, dry_run=dry_run, min_ssim=min_ssim,
                                     jobs=jobs, force=force, safe_text=safe_text,
                                     mrc_ssim=mrc_ssim, log=log)
        return str(path), "ok", saved, status
    except Skipped as exc:
        return str(path), "skipped", 0, f"пропущен: {exc}"
    except (OSError, ValueError) as exc:
        return str(path), "failed", 0, f"ошибка: {exc}"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name, value in (("quality", args.quality), ("mrc-quality", args.mrc_quality)):
        if value is not None and not 1 <= value <= 100:
            print(f"pdfopt: {name}: ожидается число от 1 до 100", file=sys.stderr)
            return 2
    mode = "lossy" if args.quality is not None else "lossless"
    min_ssim = args.quality / 100 if args.quality is not None else MIN_SSIM
    mrc_ssim = args.mrc_quality / 100 if args.mrc_quality is not None else None
    try:
        paths = discover(args.paths, args.recursive)
    except (ValueError, OSError) as exc:
        print(f"pdfopt: {exc}", file=sys.stderr)
        return 2
    if not paths:
        print("Подходящих файлов не найдено.")
        return 0
    if not ect_available():
        print("pdfopt: предупреждение: модуль ECT не собран "
              "(python3 setup.py build_ext --inplace), потоки сжимаются zlib",
              file=sys.stderr)
    if find_worker() is None:
        print("pdfopt: предупреждение: hpdf-worker не найден в PATH, "
              "Hyper Compress не используется", file=sys.stderr)

    cores = os.cpu_count() or 1
    files_at_once = min(cores, len(paths))
    per_file = max(1, cores // files_at_once)
    tasks = [(path, mode, args.dry_run, min_ssim, mrc_ssim, per_file, args.force,
              args.safe_text, args.verbose) for path in paths]
    failures = skipped = total_saved = 0

    def report(result) -> None:
        nonlocal failures, skipped, total_saved
        name, outcome, saved, text = result
        if outcome == "failed":
            failures += 1
            print(f"{name}: {text}", file=sys.stderr)
            return
        skipped += outcome == "skipped"
        total_saved += saved
        print(f"{name}: {text}" + (f", {saved} байт" if saved else ""))

    if files_at_once > 1:
        with ProcessPoolExecutor(max_workers=files_at_once) as pool:
            for result in pool.map(_process, tasks):
                report(result)
    else:
        for task in tasks:
            report(_process(task))
    print(f"Экономия: {total_saved} байт" +
          (" (оценка, файлы не изменены)" if args.dry_run else "") +
          (f"; пропущено: {skipped}" if skipped else "") +
          (f"; ошибок: {failures}" if failures else ""))
    return 1 if failures else 0
