"""Lossy re-encoding of photographic images under an SSIM budget.

For every image the search looks for the smallest JPEG whose appearance at the
SSIM render resolution stays above the threshold. Two levers are combined:
downsampling to a lower effective DPI (page geometry is read from the content
stream) and the JPEG quality (bisection instead of a fixed ladder).
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from pathlib import Path

import mozjpeg_lossless_optimization
import numpy as np
import pikepdf
from pikepdf import Name
from PIL import Image

from . import imagedata
from .pdfmodel import collect_images, filters_of, page_resources
from .render import SSIM_DPI
from .ssim import ssim
from .tools import run, which

TARGET_DPIS = (300, 225, 150)   # downsampling steps, gentlest first
QUALITY_RANGE = (22, 95)
MIN_GAIN = 0.90
SEARCH_MARGIN = 0.008           # the model of poppler's scaling is approximate
MIN_SIDE = 96
BLIND_VIEW = 2000               # longest view side when the DPI is unknown
MAX_UPSCALE = 4.0
MAX_VIEW_PIXELS = 16_000_000


# ---------------------------------------------------------------- geometry --

def effective_dpis(pdf: pikepdf.Pdf) -> dict[tuple, float]:
    """Highest resolution (dpi) at which each page-level image is shown."""
    result: dict[tuple, float] = {}
    for page in pdf.pages:
        resources = page_resources(page.obj)
        xobjects = resources.get("/XObject") if resources is not None else None
        if xobjects is None:
            continue
        stack = [(1.0, 0.0, 0.0, 1.0)]
        try:
            instructions = pikepdf.parse_content_stream(page)
        except (pikepdf.PdfError, ValueError):
            continue
        for item in instructions:
            if isinstance(item, pikepdf.ContentStreamInlineImage):
                continue
            operator = str(item.operator)
            if operator == "q":
                stack.append(stack[-1])
            elif operator == "Q" and len(stack) > 1:
                stack.pop()
            elif operator == "cm" and len(item.operands) == 6:
                a, b, c, d = (float(v) for v in item.operands[:4])
                pa, pb, pc, pd = stack[-1]
                stack[-1] = (a * pa + b * pc, a * pb + b * pd,
                             c * pa + d * pc, c * pb + d * pd)
            elif operator == "Do" and item.operands:
                obj = xobjects.get(item.operands[0])
                if obj is None or obj.get("/Subtype") != "/Image":
                    continue
                a, b, c, d = stack[-1]
                width_pt = max((a * a + b * b) ** 0.5, 1e-6)
                height_pt = max((c * c + d * d) ** 0.5, 1e-6)
                dpi = max(int(obj.get("/Width", 0)) * 72 / width_pt,
                          int(obj.get("/Height", 0)) * 72 / height_pt)
                result[obj.objgen] = max(result.get(obj.objgen, 0.0), dpi)
    return result


# ----------------------------------------------------------------- encoding --

@cache
def _mozjpeg() -> str | None:
    cjpeg = which("cjpeg")
    if cjpeg is None:
        return None
    done = run([cjpeg, "-version"])
    return cjpeg if "mozjpeg" in (done.stderr + done.stdout).lower() else None


def encode(bitmap: Image.Image, quality: int, *, progressive: bool = False) -> bytes:
    buffer = BytesIO()
    bitmap.save(buffer, "JPEG", quality=quality, optimize=True,
                progressive=progressive,
                subsampling=2 if bitmap.mode == "RGB" and quality < 90 else 0)
    return buffer.getvalue()


def tighten(data: bytes) -> bytes:
    """The same JPEG (same coefficients, so the same pixels) in fewer bytes:
    mozjpeg's optimized Huffman tables and progressive scans."""
    try:
        smaller = mozjpeg_lossless_optimization.optimize(data)
    except ValueError:
        return data
    return smaller if len(smaller) < len(data) else data


def smallest_encoding(bitmap: Image.Image, quality: int, scratch: Path) -> bytes:
    """Same pixels, fewest bytes: libjpeg's coefficients with mozjpeg's
    entropy coding (and mozjpeg's own encoder, if installed)."""
    best = tighten(encode(bitmap, quality))
    moz = _mozjpeg()
    if moz is not None:
        source = scratch / ("moz.ppm" if bitmap.mode == "RGB" else "moz.pgm")
        target = scratch / "moz.jpg"
        bitmap.save(source)
        done = run([moz, "-quality", str(quality), "-outfile", str(target), str(source)])
        if done.returncode == 0 and target.is_file() and target.stat().st_size < len(best):
            best = target.read_bytes()
    return best


def view_factor(width: int, height: int, dpi: float | None) -> float:
    """Scale from image pixels to the SSIM render resolution.

    Images shown below the render dpi are enlarged (up to 4x): poppler
    replicates pixels when enlarging, and SSIM on such a blocky enlargement is
    far more sensitive to JPEG artefacts than on the native pixels.
    """
    if dpi:
        factor = min(MAX_UPSCALE, SSIM_DPI / dpi)
    else:
        factor = min(1.0, BLIND_VIEW / max(width, height))
    return min(factor, (MAX_VIEW_PIXELS / (width * height)) ** 0.5)


def view_of(bitmap: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """The image as poppler draws it: area averaging down, replication up."""
    if bitmap.size != size:
        method = (Image.Resampling.NEAREST if size[0] > bitmap.width
                  else Image.Resampling.BOX)
        bitmap = bitmap.resize(size, method)
    return np.asarray(bitmap)


def bisect_quality(passes, low: int = QUALITY_RANGE[0],
                   high: int = QUALITY_RANGE[1]):
    """Smallest quality for which ``passes(q)`` returns data -> (q, data).

    SSIM grows with quality, so a bisection needs about six probes instead of
    walking a ladder. ``high`` is only probed when nothing below passed.
    """
    best, best_quality = None, high
    while high - low > 2:
        middle = (low + high) // 2
        data = passes(middle)
        if data is not None:
            best, best_quality, high = data, middle, middle
        else:
            low = middle
    if best is None:
        data = passes(high)
        if data is None:
            return None
        return high, data
    return best_quality, best


def search_quality(bitmap: Image.Image, reference: np.ndarray,
                   view_size: tuple[int, int], minimum: float):
    def passes(quality: int) -> bytes | None:
        data = encode(bitmap, quality)
        with Image.open(BytesIO(data)) as decoded:
            decoded.load()
            if ssim(reference, view_of(decoded, view_size)) >= minimum:
                return data
        return None

    return bisect_quality(passes)


def reencode(bitmap: Image.Image, stored: int, dpi: float | None,
             minimum: float, scratch: Path) -> tuple[bytes, tuple[int, int]] | None:
    width, height = bitmap.size
    factor = view_factor(width, height, dpi)
    view_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    reference = view_of(bitmap, view_size)
    scales = [1.0]
    if dpi:
        scales += [target / dpi for target in TARGET_DPIS
                   if target < dpi * 0.9 and min(width, height) * target / dpi >= MIN_SIDE]
    best: tuple[bytes, tuple[int, int], int] | None = None
    for scale in scales:
        size = (width, height) if scale == 1.0 else (
            max(1, round(width * scale)), max(1, round(height * scale)))
        candidate = bitmap if scale == 1.0 else bitmap.resize(size, Image.Resampling.LANCZOS)
        found = search_quality(candidate, reference, view_size,
                               min(1.0, minimum + SEARCH_MARGIN))
        if found is None:
            continue
        quality, data = found
        if best is None or len(data) < len(best[0]):
            best = (data, size, quality)
        elif scale < 1.0:
            break   # a stronger reduction no longer helps
    if best is None:
        return None
    # Re-encode the winner with the best entropy coder for those pixels.
    candidate = bitmap if best[1] == bitmap.size else bitmap.resize(
        best[1], Image.Resampling.LANCZOS)
    data = smallest_encoding(candidate, best[2], scratch)
    if len(data) >= stored * MIN_GAIN:
        return None
    return data, best[1]


# ------------------------------------------------------------------- driver --

@dataclass
class Change:
    """An image rewritten in place, with what is needed to undo it."""

    obj: pikepdf.Object
    pages: set[int]                 # zero-based pages showing the image
    entries: dict
    raw: bytes


def revert(change: Change) -> None:
    obj = change.obj
    for key in [k for k in obj if k != "/Length"]:
        del obj[key]
    for key, value in change.entries.items():
        if key not in ("/Length", "/Filter", "/DecodeParms"):
            obj[key] = value
    obj.write(change.raw, filter=change.entries.get("/Filter"),
              decode_parms=change.entries.get("/DecodeParms"))


@dataclass
class Job:
    objgen: tuple
    jpeg: bytes | None
    rows: np.ndarray | None
    width: int
    height: int
    ncomp: int
    stored: int
    dpi: float | None


def _photographic(rows: np.ndarray, ncomp: int) -> bool:
    """Flate pictures with many distinct colours are worth a JPEG."""
    flat = rows.reshape(-1, ncomp) if rows.shape[1] % ncomp == 0 else None
    if flat is None or len(flat) < 4096:
        return False
    step = max(1, len(flat) // 65536)
    sample = np.ascontiguousarray(flat[::step])
    return len(np.unique(sample.view(np.dtype((np.void, sample.dtype.itemsize * ncomp))))) > 1500


def _eligible_image(obj) -> bool:
    if obj.get("/BitsPerComponent") != 8 or "/Decode" in obj or "/ImageMask" in obj:
        return False
    if "/DecodeParms" in obj and filters_of(obj) == ["/DCTDecode"]:
        return False
    if isinstance(obj.get("/Mask"), pikepdf.Array):
        return False
    ncomp = imagedata.components(obj.get("/ColorSpace"))
    cs = obj.get("/ColorSpace")
    if ncomp not in (1, 3) or (isinstance(cs, pikepdf.Array) and str(cs[0]) == "/Indexed"):
        return False
    return min(int(obj.get("/Width", 0)), int(obj.get("/Height", 0))) >= MIN_SIDE


def _extract(obj, dpi: float | None) -> Job | None:
    try:
        width, height = int(obj.Width), int(obj.Height)
        ncomp = imagedata.components(obj.get("/ColorSpace"))
        stored = len(obj.read_raw_bytes())
        filters = filters_of(obj)
        if filters in (["/DCTDecode"], ["/FlateDecode", "/DCTDecode"]):
            data = imagedata.jpeg_bytes(obj)
            if data is None:
                return None
            return Job(obj.objgen, data, None, width, height, ncomp, stored, dpi)
        if any(f not in imagedata.GENERAL_FILTERS for f in filters) or stored < 20_000:
            return None
        rows = imagedata.raw_samples(obj, width, height, ncomp, 8)
        if rows is None or not _photographic(rows, ncomp):
            return None
        return Job(obj.objgen, None, rows, width, height, ncomp, stored, dpi)
    except (OSError, ValueError, KeyError, TypeError, pikepdf.PdfError):
        return None


def _run(job: Job, minimum: float, scratch: Path):
    try:
        bitmap = imagedata.bitmap_from(job.jpeg, job.rows, job.width, job.height, job.ncomp)
        if bitmap is None:
            return None
        return reencode(bitmap, job.stored, job.dpi, minimum, scratch)
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError):
        return None


def optimize_jpegs(pdf: pikepdf.Pdf, scratch: Path, minimum: float,
                   jobs: int) -> tuple[Counter, list[Change]]:
    """Re-encode eligible images in place.

    Returns counters for the log and one undoable ``Change`` per image.
    """
    dpis = effective_dpis(pdf)
    entries = []
    for ref in collect_images(pdf).values():
        if _eligible_image(ref.obj):
            entries.append(ref)
    counts: Counter = Counter()
    changes: list[Change] = []
    window = max(2, jobs * 2)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for start in range(0, len(entries), window):
            batch = []
            for index, ref in enumerate(entries[start:start + window]):
                obj = ref.obj
                job = _extract(obj, dpis.get(obj.objgen))
                if job is None:
                    continue
                directory = scratch / f"jpeg-{start + index}"
                directory.mkdir(exist_ok=True)
                batch.append((obj, job, directory, ref.pages))
            results = pool.map(lambda b: _run(b[1], minimum, b[2]), batch)
            for (obj, job, _, pages), result in zip(batch, results):
                if result is None:
                    continue
                data, (width, height) = result
                changes.append(Change(obj, pages, {k: obj[k] for k in obj},
                                      bytes(obj.read_raw_bytes())))
                obj.write(data, filter=Name.DCTDecode, decode_parms=None)
                obj.Width, obj.Height = width, height
                obj.BitsPerComponent = 8
                counts["resized" if (width, height) != (job.width, job.height)
                       else "recoded"] += 1
    return counts, changes
