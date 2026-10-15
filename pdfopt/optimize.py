"""PDF optimization via Hyper Compress, with local fallbacks."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject
from PIL import Image

from .mrc import add_layers, restore


HYPER_TOKENS = {
    "lossless": ("3=0", "10=1", "13=1", "14=1", "16=1", "17=1"),
    "lossy": (
        "1=150", "2=50", "3=4", "8=1", "18=1", "7=1", "63=10",
        "6=1", "10=1", "13=1", "14=1", "16=1", "17=1", "40=1",
    ),
}
TIMEOUT = 600
MIN_SSIM = 0.96
SSIM_DPI = 200
# Ghostscript's QFactor grows as JPEG compression becomes stronger.
GHOSTSCRIPT_PROFILES = ((300, 0.35), (240, 0.50), (200, 0.65))
JPEG_QUALITIES = (35, 45, 60, 75, 90)


def _has_signature(path: Path, reader: PdfReader) -> bool:
    root = reader.trailer["/Root"]
    if root.get("/Perms"):
        return True
    tail = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if b"/ByteRange" in tail + chunk:
                return True
            tail = chunk[-16:]
    return False


def _page_snapshot(reader: PdfReader) -> tuple:
    pages = []
    for page in reader.pages:
        annots = page.get("/Annots")
        if annots:
            annots = annots.get_object()
        box = tuple(round(float(v), 2) for v in page.mediabox)
        crop = tuple(round(float(v), 2) for v in page.cropbox)
        text = page.extract_text() or ""
        pages.append((box, crop, page.rotation, len(annots or []),
                      hashlib.sha256(text.encode("utf-8")).digest()))
    metadata = tuple(sorted((str(k), str(v)) for k, v in (reader.metadata or {}).items()))
    return tuple(pages), metadata


def _pdf_snapshot(path: Path) -> tuple:
    with path.open("rb") as stream:
        reader = PdfReader(stream, strict=True)
        if reader.is_encrypted:
            raise ValueError("PDF: зашифрованный файл пока не поддерживается")
        if _has_signature(path, reader):
            raise ValueError("PDF: документ с электронной подписью не изменён")
        pages, metadata = _page_snapshot(reader)
        root = reader.trailer["/Root"]
        interactive = ("/AcroForm" in root or "/Names" in root or
                       any(page[3] for page in pages))
        return pages, metadata, interactive


def _check_candidate(path: Path, baseline: tuple, lossless: bool) -> None:
    with path.open("rb") as stream:
        reader = PdfReader(stream, strict=True)
        if reader.is_encrypted:
            raise ValueError("PDF: результат зашифрован")
        pages, metadata = _page_snapshot(reader)
    if pages != baseline[0]:
        raise ValueError("PDF: изменились страницы, текст, геометрия или аннотации")
    if lossless and metadata != baseline[1]:
        raise ValueError("PDF: изменились метаданные в режиме без потерь")


def _find_worker(explicit: Path | None) -> tuple[Path | None, bool]:
    if explicit is not None:
        if not explicit.is_file() or not os.access(explicit, os.X_OK):
            raise ValueError(f"PDF: hpdf-worker не найден или не исполняется: {explicit}")
        return explicit, True
    for key in ("PDFOPT_HYPER_WORKER", "BIBLIOPTER_HYPER_WORKER", "HYPER_DRV"):
        if value := os.environ.get(key):
            worker = Path(value).expanduser().absolute()
            if not worker.is_file() or not os.access(worker, os.X_OK):
                raise ValueError(f"PDF: {key} указывает на недоступный hpdf-worker: {worker}")
            return worker, True
    bundled = (Path(__file__).resolve().parent.parent / "third_party" /
               "bentopdf-hyper-compress" / "cli" / "prebuilt" / "hpdf-worker")
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return bundled, False
    found = shutil.which("hpdf-worker")
    return (Path(found), False) if found else (None, False)


def _make_pypdf_candidate(source: Path, target: Path) -> None:
    with source.open("rb") as stream:
        reader = PdfReader(stream, strict=True)
        writer = PdfWriter(clone_from=reader)
        for page in writer.pages:
            page.compress_content_streams(level=9)
        writer.compress_identical_objects(remove_duplicates=True,
                                          remove_unreferenced=False)
        with target.open("wb") as output:
            writer.write(output)


def _image_objects(page):
    resources = [page.get("/Resources")]
    seen = set()
    while resources:
        resource = resources.pop()
        if resource is None:
            continue
        xobjects = resource.get_object().get("/XObject")
        if xobjects is None:
            continue
        for reference in xobjects.get_object().values():
            obj = reference.get_object()
            key = (id(reference.pdf), reference.idnum) if hasattr(reference, "idnum") else id(obj)
            if key in seen:
                continue
            seen.add(key)
            if obj.get("/Subtype") == "/Image":
                yield obj
            elif obj.get("/Subtype") == "/Form":
                resources.append(obj.get("/Resources"))


def _make_jpeg_candidate(source: Path, target: Path, scratch: Path,
                         min_ssim: float | None = None) -> None:
    """Re-encode simple JPEG XObjects without touching the page content streams."""
    if min_ssim is None:
        min_ssim = MIN_SSIM
    original_image = scratch / "image-original.jpg"
    trial_image = scratch / "image-trial.jpg"
    changed = 0
    visited = set()
    with source.open("rb") as stream:
        writer = PdfWriter(clone_from=PdfReader(stream, strict=True))
        for page in writer.pages:
            for obj in _image_objects(page):
                reference = obj.indirect_reference
                key = (id(reference.pdf), reference.idnum) if reference else id(obj)
                if key in visited:
                    continue
                visited.add(key)
                filters = obj.get("/Filter")
                if (filters != "/DCTDecode" and
                    list(filters or []) != ["/FlateDecode", "/DCTDecode"]):
                    continue
                if (
                    obj.get("/ColorSpace") not in ("/DeviceRGB", "/DeviceGray") or
                    obj.get("/BitsPerComponent") != 8 or
                    any(key in obj for key in ("/SMask", "/Mask", "/Decode", "/DecodeParms"))):
                    continue
                source_bytes = obj.get_data()
                original_image.write_bytes(source_bytes)
                try:
                    with Image.open(BytesIO(source_bytes)) as bitmap:
                        if bitmap.mode not in ("RGB", "L"):
                            continue
                        for quality in JPEG_QUALITIES:
                            buffer = BytesIO()
                            bitmap.save(buffer, "JPEG", quality=quality, optimize=True)
                            encoded = buffer.getvalue()
                            if len(encoded) >= len(source_bytes):
                                continue
                            trial_image.write_bytes(encoded)
                            if _page_ssim(original_image, trial_image) >= min_ssim:
                                obj._data = encoded
                                obj[NameObject("/Filter")] = NameObject("/DCTDecode")
                                changed += 1
                                break
                except (OSError, ValueError):
                    continue
        if not changed:
            raise ValueError(f"не найдено JPEG-изображений при SSIM ≥ {min_ssim:.2f}")
        with target.open("wb") as output:
            writer.write(output)


def _make_hyper_candidate(worker: Path, source: Path, target: Path,
                          mode: str) -> None:
    completed = subprocess.run(
        [str(worker), str(source), str(target), *HYPER_TOKENS[mode]],
        capture_output=True, text=True, check=False, timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0 or not target.is_file():
        details = (completed.stderr or completed.stdout).strip().splitlines()
        raise ValueError("Hyper Compress: " + (details[-1] if details else
                         f"код выхода {completed.returncode}"))
    if "SIGNED_SKIP" in completed.stderr:
        raise ValueError("PDF: Hyper Compress обнаружил электронную подпись")


def _make_ghostscript_candidate(source: Path, target: Path,
                                dpi: int, qfactor: float) -> None:
    ghostscript = shutil.which("gs")
    if not ghostscript:
        raise ValueError("Ghostscript не найден")
    completed = subprocess.run(
        [ghostscript, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-q",
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
         "-f", str(source)],
        capture_output=True, text=True, check=False, timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0 or not target.is_file():
        details = (completed.stderr or completed.stdout).strip().splitlines()
        raise ValueError("Ghostscript: " + (details[-1] if details else
                         f"код выхода {completed.returncode}"))


def _render_pages(pdf: Path, prefix: Path, expected: int) -> list[Path]:
    completed = subprocess.run(
        [shutil.which("pdftoppm"), "-r", str(SSIM_DPI), "-cropbox",
         "-png", str(pdf), str(prefix)],
        capture_output=True, text=True, check=False, timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
    )
    pages = sorted(prefix.parent.glob(prefix.name + "-*.png"))
    if completed.returncode != 0 or len(pages) != expected:
        raise ValueError("PDF: не удалось отрисовать все страницы для SSIM")
    return pages


def _page_ssim(reference: Path, candidate: Path) -> float:
    completed = subprocess.run(
        [shutil.which("ffmpeg"), "-hide_banner", "-nostdin", "-nostats", "-i", str(reference),
         "-i", str(candidate), "-filter_complex",
         "[0:v]format=gbrp[a];[1:v]format=gbrp[b];[a][b]ssim",
         "-frames:v", "1", "-f", "null", "-"],
        capture_output=True, text=True, check=False, timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
    )
    match = re.search(r"\bAll:([0-9]+(?:\.[0-9]+)?)", completed.stderr)
    if completed.returncode != 0 or match is None:
        raise ValueError("PDF: FFmpeg не смог вычислить SSIM")
    return float(match.group(1))


def _visual_similarity(reference: list[Path], candidate: Path,
                       scratch: Path, min_ssim: float | None = None) -> bool:
    if min_ssim is None:
        min_ssim = MIN_SSIM
    rendered = scratch / "rendered"
    rendered.mkdir()
    try:
        pages = _render_pages(candidate, rendered / "page", len(reference))
        return all(_page_ssim(original, result) >= min_ssim
                   for original, result in zip(reference, pages))
    finally:
        shutil.rmtree(rendered)


def _render_page(pdf: Path, page: int, output: Path) -> None:
    if output.suffix not in (".png", ".ppm"):
        raise ValueError("PDF: формат страницы должен быть PNG или PPM")
    completed = subprocess.run(
        [shutil.which("pdftoppm"), "-f", str(page), "-l", str(page),
         "-singlefile", "-r", str(SSIM_DPI), "-cropbox",
         *(["-png"] if output.suffix == ".png" else []),
         str(pdf), str(output.with_suffix(""))],
        capture_output=True, text=True, check=False, timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0 or not output.is_file():
        raise ValueError(f"PDF: не удалось отрисовать страницу {page} для MRC")


def _make_mrc_candidate(source: Path, target: Path, scratch: Path,
                        min_ssim: float | None = None) -> None:
    if min_ssim is None:
        min_ssim = MIN_SSIM
    with source.open("rb") as stream:
        writer = PdfWriter(clone_from=PdfReader(stream, strict=True))
        records = add_layers(writer, scratch, _page_ssim, min_ssim)
        if not records:
            raise ValueError("нет страниц, подходящих для слойного сжатия MRC")
        with target.open("wb") as output:
            writer.write(output)
        passed = 0
        rejected = 0
        def check(record) -> bool:
            page = record[0]
            original = scratch / f"mrc-page-{page}-original.ppm"
            result = scratch / f"mrc-page-{page}-result.ppm"
            try:
                _render_page(source, page, original)
                _render_page(target, page, result)
                return _page_ssim(original, result) >= min_ssim
            except (ValueError, subprocess.TimeoutExpired):
                return False
            finally:
                original.unlink(missing_ok=True)
                result.unlink(missing_ok=True)

        workers = min(4, os.cpu_count() or 1) if len(records) >= 16 else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            checks = pool.map(check, records)
            for index, (record, valid) in enumerate(zip(records, checks), start=1):
                if valid:
                    passed += 1
                else:
                    restore(record)
                    rejected += 1
                if len(records) >= 50 and index % 50 == 0:
                    print(f"PDF MRC: проверено {index}/{len(records)} страниц",
                          file=sys.stderr, flush=True)
        if not passed:
            raise ValueError(f"ни одна страница MRC не прошла SSIM ≥ {min_ssim:.2f}")
        if rejected:
            writer.compress_identical_objects(remove_duplicates=False,
                                              remove_unreferenced=True)
            with target.open("wb") as output:
                writer.write(output)
        if len(writer.pages) >= 50:
            print(f"PDF MRC: принято {passed}, отклонено {rejected} страниц",
                  file=sys.stderr, flush=True)


def optimize_pdf(path: Path, mode: str, dry_run: bool,
                 worker_path: Path | None = None,
                 min_ssim: float = MIN_SSIM) -> tuple[str, int]:
    """Choose the smallest verified PDF and replace the source atomically."""
    source_stat = path.stat()
    try:
        baseline = _pdf_snapshot(path)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PDF: не удалось прочитать исходник: {exc}") from exc
    worker, explicit = _find_worker(worker_path)
    if mode == "lossy":
        if shutil.which("ffmpeg") is None:
            raise ValueError("PDF: для проверки SSIM нужен ffmpeg")
        if shutil.which("pdftoppm") is None:
            raise ValueError("PDF: для проверки MRC нужен pdftoppm")
        if worker is not None and shutil.which("pdftoppm") is None:
            raise ValueError("PDF: для проверки Hyper Compress нужен pdftoppm")
    errors = []
    best: tuple[Path, str, int] | None = None
    with tempfile.TemporaryDirectory(prefix=".pdfopt-", dir=path.parent) as directory:
        scratch = Path(directory)
        engines = [("pypdf", _make_pypdf_candidate)]
        if mode == "lossy":
            engines.append((f"JPEG SSIM≥{min_ssim:.2f}",
                            lambda src, dst: _make_jpeg_candidate(src, dst, scratch, min_ssim)))
            engines.append((f"MRC SSIM≥{min_ssim:.2f}",
                            lambda src, dst: _make_mrc_candidate(src, dst, scratch, min_ssim)))
        if worker is not None:
            engines.append(("Hyper Compress", lambda src, dst: _make_hyper_candidate(worker, src, dst, mode)))
        has_text = any(page[4] != hashlib.sha256(b"").digest() for page in baseline[0])
        if (mode == "lossy" and not baseline[2] and not has_text and
                shutil.which("gs") and shutil.which("pdftoppm")):
            for dpi, qfactor in GHOSTSCRIPT_PROFILES:
                engines.append((f"Ghostscript {dpi} dpi/Q{qfactor}",
                                lambda src, dst, dpi=dpi, qfactor=qfactor:
                                _make_ghostscript_candidate(src, dst, dpi, qfactor)))
        reference = None
        for index, (name, engine) in enumerate(engines):
            candidate = scratch / f"candidate-{index}.pdf"
            try:
                engine(path, candidate)
                if not candidate.is_file() or candidate.stat().st_size == 0:
                    raise ValueError("пустой результат")
                size = candidate.stat().st_size
                if size < source_stat.st_size and (best is None or size < best[2]):
                    _check_candidate(candidate, baseline, mode == "lossless")
                    if (mode == "lossy" and name != "pypdf" and
                            not name.startswith(("JPEG SSIM≥", "MRC SSIM≥"))):
                        if reference is None:
                            reference = _render_pages(path, scratch / "original", len(baseline[0]))
                        if not _visual_similarity(reference, candidate, scratch, min_ssim):
                            continue
                    best = (candidate, name, size)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                if name == "Hyper Compress" and explicit:
                    raise ValueError(errors[-1]) from exc
        if best is None:
            if len(errors) == len(engines):
                raise ValueError("PDF: " + "; ".join(errors))
            return "без изменений", 0
        candidate, name, size = best
        saving = source_stat.st_size - size
        if dry_run:
            return f"возможная экономия ({name})", saving
        current = path.stat()
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
            source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns
        ):
            raise ValueError("исходный файл изменился во время обработки")
        shutil.copystat(path, candidate)
        os.replace(candidate, path)
        return f"сжато ({name})", saving
