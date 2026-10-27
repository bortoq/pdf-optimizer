"""The lossless pipeline: one parse, several passes, one save.

Passes (each optional and independent):

* structure: drop page thumbnails, merge identical streams;
* images: RGB->Gray, 8->1 bit, PNG predictors, Group 4/Flate -> JBIG2 generic;
* JPEG: mozjpeg Huffman optimization / progressive scans (pixel-checked);
* fonts: embedded TrueType fonts cut to the glyphs in use (pixel-checked);
* streams: ECT -9 instead of zlib, then qpdf object/xref streams on save.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import struct
import tempfile
import zlib

import mozjpeg_lossless_optimization
import numpy as np
import pikepdf
from pikepdf import Name
from PIL import Image

from . import fonts, imagedata
from .pdfmodel import collect_images, filters_of, iter_streams, open_pdf
from .tools import ToolError, run, which

try:   # the bundled ECT compressor (build: python setup.py build_ext --inplace)
    from . import _ectdeflate
except ImportError:  # pragma: no cover - depends on the environment
    _ectdeflate = None

pikepdf.settings.set_flate_compression_level(9)

MIN_GAIN = 0.98            # an image rewrite must save at least 2 %
MIN_JBIG2_PIXELS = 100_000
ECT_BUDGET = 256 << 20     # bytes of raw data re-deflated per document
ECT_FAST_ABOVE = 4 << 20   # streams above this get ECT level 6 instead of 9


@dataclass
class Report:
    """What a build touched, for targeted verification."""

    visual_pages: set[int]
    notes: list[str]


# --------------------------------------------------------------- structure --

def drop_thumbnails(pdf: pikepdf.Pdf) -> int:
    removed = 0
    for page in pdf.pages:
        if "/Thumb" in page.obj:
            del page.obj["/Thumb"]
            removed += 1
    return removed


_CONTAINERS = (pikepdf.ObjectType.array, pikepdf.ObjectType.dictionary)


def _text(value) -> bytes | str:
    return value.unparse() if hasattr(value, "unparse") else repr(value)


def _stream_key(stream) -> tuple:
    entries = tuple(sorted((str(key), _text(value))
                           for key, value in stream.items() if key != "/Length"))
    return hashlib.sha1(stream.read_raw_bytes()).digest(), entries


def merge_identical_streams(pdf: pikepdf.Pdf) -> int:
    """Point every reference to a duplicate stream at its first copy."""
    canonical: dict[tuple, pikepdf.Object] = {}
    redirect: dict[tuple, pikepdf.Object] = {}
    for stream in iter_streams(pdf):
        if stream.get("/Type") in ("/XRef", "/ObjStm"):
            continue
        key = _stream_key(stream)
        first = canonical.setdefault(key, stream)
        if first.objgen != stream.objgen:
            redirect[stream.objgen] = first
    if not redirect:
        return 0

    def fix(container) -> None:
        if isinstance(container, pikepdf.Array):
            pairs = list(enumerate(container))
        elif isinstance(container, (pikepdf.Dictionary, pikepdf.Stream)):
            pairs = list(container.items())
        else:
            return
        for key, value in pairs:
            if not isinstance(value, pikepdf.Object):
                continue
            if value.is_indirect:
                target = redirect.get(value.objgen)
                if target is not None:
                    container[key] = target
            elif value._type_code in _CONTAINERS:
                fix(value)

    for obj in list(pdf.objects):
        fix(obj)
    fix(pdf.trailer)
    return len(redirect)


# -------------------------------------------------------------- JPEG pass --

def _same_pixels(first: bytes, second: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(first)) as a, Image.open(io.BytesIO(second)) as b:
            return (a.mode == b.mode and a.size == b.size and
                    a.tobytes() == b.tobytes())
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def optimize_jpeg(data: bytes) -> bytes | None:
    """The same JPEG coefficients in fewer bytes, or None.

    mozjpeg's ``jpegtran`` (optimized Huffman tables, searched progressive
    scans; no extra markers, like ``-copy none``) beats libjpeg's by 0.7 % on
    photo scans and up to 28 % on smooth images. The pixels are compared
    after decoding all the same.
    """
    try:
        encoded = mozjpeg_lossless_optimization.optimize(data)
    except ValueError:
        return None
    if len(encoded) >= len(data) or not _same_pixels(data, encoded):
        return None
    return encoded


# ------------------------------------------------------------- image tasks --

@dataclass
class ImageInput:
    """Everything a worker needs, extracted on the main thread (qpdf objects
    are not shared between threads)."""

    kind: str                       # "jpeg" or "raw"
    width: int = 0
    height: int = 0
    bits: int = 8
    ncomp: int = 1
    family: str | None = None       # "gray" / "rgb" for device spaces
    stored: int = 0
    rows: np.ndarray | None = None
    jpeg: bytes | None = None
    can_threshold: bool = True      # may an exact black/white picture go 1-bit
    ccitt_source: bool = False
    can_gray: bool = True           # RGB->Gray keeps the dictionary valid (no /Decode)


@dataclass
class Replacement:
    data: bytes
    filter: pikepdf.Name
    parms: pikepdf.Object | None
    updates: dict
    note: str


def _flate(data: bytes) -> bytes:
    return zlib.compress(data, 9)


def _bilevel_rows(obj, width, height):
    """Packed sample rows of a 1-bit image, or None if unsupported."""
    filters = filters_of(obj)
    if filters == ["/CCITTFaxDecode"]:
        parms = obj.get("/DecodeParms")
        if isinstance(parms, pikepdf.Array):
            parms = parms[0] if len(parms) == 1 else None
        if parms is None or int(parms.get("/K", 0)) >= 0:
            return None
        if (int(parms.get("/Columns", 1728)) != width or
                parms.get("/EncodedByteAlign", False) or
                parms.get("/EndOfLine", False)):
            return None
        return imagedata.g4_bitmap(bytes(obj.read_raw_bytes()), width, height,
                                   bool(parms.get("/BlackIs1", False)))
    return imagedata.raw_samples(obj, width, height, 1, 1)


def extract(kind: str, obj) -> ImageInput | None:
    """Read one image into plain Python/numpy data (main thread only)."""
    try:
        stored = len(obj.read_raw_bytes())
        if kind == "jpeg":
            data = imagedata.jpeg_bytes(obj)
            return ImageInput("jpeg", jpeg=data, stored=stored) if data else None
        width, height = int(obj.get("/Width", 0)), int(obj.get("/Height", 0))
        if width <= 0 or height <= 0:
            return None
        mask = bool(obj.get("/ImageMask", False))
        bits = int(obj.get("/BitsPerComponent", 1 if mask else 0))
        colorspace = obj.get("/ColorSpace")
        family = imagedata.device_family(colorspace)
        ncomp = 1 if mask else imagedata.components(colorspace)
        if ncomp is None or isinstance(obj.get("/Mask"), pikepdf.Array):
            return None
        if bits == 1 and ncomp == 1 and (mask or family == "gray"):
            rows = _bilevel_rows(obj, width, height)
        elif bits == 8:
            rows = imagedata.raw_samples(obj, width, height, ncomp, 8)
        else:
            return None
        if rows is None:
            return None
        return ImageInput("raw", width, height, bits, ncomp, family, stored, rows,
                          can_threshold="/Matte" not in obj,
                          ccitt_source=filters_of(obj) == ["/CCITTFaxDecode"],
                          can_gray="/Decode" not in obj)
    except (OSError, ValueError, KeyError, TypeError, pikepdf.PdfError):
        return None


def _jbig2_generic(rows, width, height, scratch: Path) -> bytes | None:
    jbig2 = which("jbig2")
    if jbig2 is None or width * height < MIN_JBIG2_PIXELS:
        return None
    source = scratch / "page.png"
    target = scratch / "page.jb2"
    source.write_bytes(imagedata.to_png(rows, width, height))
    done = run([jbig2, "-p", str(source)], stdout_to=target)
    if done.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        return None
    return target.read_bytes()


def _predicted(rows: np.ndarray, bits: int, ncomp: int, width: int):
    """Smaller of plain Flate and PNG-predicted Flate, with decode parms."""
    plain = _flate(rows.tobytes())
    bpp = max(1, ncomp * bits // 8)
    predicted = _flate(imagedata.png_predict(rows, bpp))
    if len(predicted) < len(plain):
        parms = pikepdf.Dictionary(Predictor=15, Colors=ncomp,
                                   BitsPerComponent=bits, Columns=width)
        return predicted, parms
    return plain, None


def rewrite(item: ImageInput, scratch: Path, jbig2_ok: bool) -> Replacement | None:
    """Pick the smallest equivalent encoding of one image (thread-safe)."""
    try:
        if item.kind == "jpeg":
            encoded = optimize_jpeg(item.jpeg)
            if encoded is None:
                return None
            return Replacement(encoded, Name.DCTDecode, None, {}, "jpeg")
        return _rewrite_raw(item, scratch, jbig2_ok)
    except (OSError, ValueError, ToolError, MemoryError, Image.DecompressionBombError):
        return None


def _rewrite_raw(item: ImageInput, scratch: Path, jbig2_ok: bool) -> Replacement | None:
    rows, bits, ncomp, width, height = (item.rows, item.bits, item.ncomp,
                                        item.width, item.height)
    updates: dict = {}
    notes: list[str] = []
    family = item.family
    if bits == 8 and family == "rgb" and ncomp == 3 and item.can_gray:
        pixels = rows.reshape(height, width, 3)
        if np.array_equal(pixels[:, :, 0], pixels[:, :, 1]) and \
                np.array_equal(pixels[:, :, 1], pixels[:, :, 2]):
            rows = np.ascontiguousarray(pixels[:, :, 0])
            ncomp, family = 1, "gray"
            updates["/ColorSpace"] = Name.DeviceGray
            notes.append("gray")
    if (bits == 8 and ncomp == 1 and family == "gray" and item.can_threshold and
            np.isin(rows, (0, 255)).all()):
        rows = imagedata.pack_bits(rows == 255)
        bits = 1
        updates["/BitsPerComponent"] = 1
        notes.append("1bit")
    candidates = []
    if bits != 1 or not item.ccitt_source:
        data, parms = _predicted(rows, bits, ncomp, width)
        candidates.append((data, Name.FlateDecode, parms))
    if bits == 1 and jbig2_ok:
        coded = _jbig2_generic(rows, width, height, scratch)
        if coded is not None:
            candidates.append((coded, Name.JBIG2Decode, None))
            notes.append("jbig2")
    if not candidates:
        return None
    data, filter_name, parms = min(candidates, key=lambda c: len(c[0]))
    if len(data) >= item.stored * MIN_GAIN:
        return None
    return Replacement(data, filter_name, parms, updates,
                       "+".join(notes) or "predictor")


def _apply(obj, result: Replacement) -> None:
    obj.write(result.data, filter=result.filter, decode_parms=result.parms)
    for key, value in result.updates.items():
        obj[key] = value


def optimize_images(pdf: pikepdf.Pdf, scratch: Path, jobs: int, *,
                    transforms: bool, jbig2: bool) -> tuple[set[int], Counter]:
    """Rewrite images in place.

    Returns the pages that need a visual check and a count of rewrites by kind.
    """
    tasks: dict[tuple, list] = {}   # objgen -> [kind, obj, pages]

    def add(kind, obj, pages):
        entry = tasks.setdefault(obj.objgen, [kind, obj, set()])
        entry[2].update(pages)

    for ref in collect_images(pdf).values():
        obj = ref.obj
        filters = filters_of(obj)
        if filters in (["/DCTDecode"], ["/FlateDecode", "/DCTDecode"]):
            if obj.get("/BitsPerComponent") == 8 and "/DecodeParms" not in obj:
                add("jpeg", obj, ref.pages)
        elif transforms and not any(f in ("/DCTDecode", "/JPXDecode", "/JBIG2Decode")
                                    for f in filters):
            add("raw", obj, ref.pages)
        for extra in ("/SMask", "/Mask"):
            child = obj.get(extra)
            if transforms and isinstance(child, pikepdf.Stream):
                add("raw", child, ref.pages)

    visual: set[int] = set()
    counts: Counter = Counter()
    entries = list(tasks.values())
    window = max(4, jobs * 2)   # bounds the decoded pixels held in memory
    with tempfile.TemporaryDirectory(dir=scratch) as base, \
            ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for start in range(0, len(entries), window):
            batch = []
            for offset, (kind, obj, pages) in enumerate(entries[start:start + window]):
                item = extract(kind, obj)
                if item is None:
                    continue
                directory = Path(base) / f"{start + offset}"
                directory.mkdir()
                batch.append((kind, obj, pages, item, directory))
            results = pool.map(lambda b: rewrite(b[3], b[4], jbig2), batch)
            for (kind, obj, pages, _, _), result in zip(batch, results):
                if result is not None:
                    _apply(obj, result)
                    counts[result.note] += 1
                    if kind == "raw":
                        visual.update(pages)
    return visual, counts


# --------------------------------------------------------------------- ECT --

def ect_available() -> bool:
    return _ectdeflate is not None


def ect_mode(size: int) -> int:
    """ECT level for ``size`` raw bytes: 9, or 6 for big streams, which is
    4-8 times faster and only 0.03-0.3 % larger (a 25 MB image: 38 s -> ~5 s)."""
    return 9 if size <= ECT_FAST_ABOVE else 6


def deflate_zlib(data: bytes, mode: int = 9) -> bytes:
    """A zlib stream whose Deflate part is made like ``ect -<mode>``."""
    return (b"\x78\xda" + _ectdeflate.deflate(data, mode) +
            struct.pack(">I", zlib.adler32(data)))


def ect_streams(pdf: pikepdf.Pdf, jobs: int) -> bool:
    """Re-deflate Flate/unfiltered streams with ECT; True if it ran."""
    if _ectdeflate is None:
        return False
    items = []
    budget = ECT_BUDGET
    for stream in iter_streams(pdf):
        if stream.get("/Type") in ("/XRef", "/ObjStm", "/Metadata"):
            continue
        filters = filters_of(stream)
        if filters not in ([], ["/FlateDecode"]):
            continue
        raw = bytes(stream.read_raw_bytes())
        try:
            if filters:   # inflate no further than the budget left (no bombs)
                inflate = zlib.decompressobj()
                data = inflate.decompress(raw, budget + 1)
                if inflate.unconsumed_tail:
                    continue
            else:
                data = raw
        except zlib.error:
            continue
        if len(data) < 64 or len(data) > budget:
            continue
        budget -= len(data)
        items.append((stream, data, len(raw) if filters else None))

    def pack(item):
        return deflate_zlib(item[1], ect_mode(len(item[1])))

    # The compressor releases the GIL, so threads scale across cores.
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        packed = list(pool.map(pack, items))
    for (stream, _, old), new in zip(items, packed):
        if old is not None and len(new) >= old:
            continue
        stream.write(new, filter=Name.FlateDecode,
                     decode_parms=stream.get("/DecodeParms"))
    return True


# ------------------------------------------------------------------- build --

def save(pdf: pikepdf.Pdf, target: Path, *, deflated: bool) -> None:
    pdf.save(target, object_stream_mode=pikepdf.ObjectStreamMode.generate,
             compress_streams=True, recompress_flate=not deflated,
             fix_metadata_version=False, deterministic_id=True)


def build(source: Path, target: Path, scratch: Path, jobs: int = 1, *,
          transforms: bool = True, jbig2: bool = True,
          ect: bool = True) -> Report:
    """Run the whole lossless pipeline from ``source`` into ``target``."""
    notes: list[str] = []
    with open_pdf(source) as pdf:
        if drop_thumbnails(pdf):
            notes.append("thumbs")
        if merge_identical_streams(pdf):
            notes.append("dedupe")
        visual, counts = optimize_images(pdf, scratch, jobs,
                                         transforms=transforms, jbig2=jbig2)
        notes.extend(f"{name}×{count}" for name, count in sorted(counts.items()))
        if transforms:
            font_pages, cut = fonts.subset_fonts(pdf)
            if cut:
                visual |= font_pages
                notes.append(f"fonts×{cut}")
        deflated = ect_streams(pdf, jobs) if ect else False
        save(pdf, target, deflated=deflated)
    return Report(visual, notes)
