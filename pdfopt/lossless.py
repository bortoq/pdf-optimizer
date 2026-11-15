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
from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
import re
import struct
import tempfile
import threading
import zlib

import mozjpeg_lossless_optimization
import numpy as np
import pikepdf
from pikepdf import Name
from PIL import Image

from . import fonts, imagedata
from .pdfmodel import collect_images, filters_of, iter_streams, open_pdf, pdfa_part
from .progress import Callback, nothing
from .tools import ToolError, run, which

try:   # the bundled ECT compressor (build: python setup.py build_ext --inplace)
    from . import _ectdeflate  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on the environment
    _ectdeflate = None

pikepdf.settings.set_flate_compression_level(9)

MIN_GAIN = 0.98            # an image rewrite must save at least 2 %
MIN_JBIG2_PIXELS = 100_000
# Most colours an image may have to be stored as indices: 256 fit a byte; a
# gray image gains only when its indices are narrower than its samples.
PALETTE_LIMIT = {1: 16, 3: 256}
ECT_BUDGET = 256 << 20     # bytes of raw data re-deflated per document
ECT_FAST_ABOVE = 4 << 20   # streams above this get ECT level 6 instead of 9


@dataclass
class Report:
    """What a build touched, for targeted verification."""

    visual_pages: set[int]
    notes: list[str]
    # Font programs that were changed -> zero-based pages showing them, so a
    # failed rendering check can drop just the fonts of the failing pages.
    fonts: dict[tuple, set[int]] = field(default_factory=dict)


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
        pairs: list
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
    can_index: bool = False         # may become indices into a colour table


@dataclass
class Replacement:
    data: bytes
    filter: pikepdf.Name
    parms: pikepdf.Object | None
    updates: dict
    note: str
    table: bytes | None = None      # colour table: the image becomes Indexed


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


def _base_space(colorspace) -> bool:
    """A colour space that may be the base of an Indexed one."""
    return isinstance(colorspace, pikepdf.Name) or (
        isinstance(colorspace, pikepdf.Array) and len(colorspace) > 0
        and str(colorspace[0]) not in ("/Indexed", "/Pattern"))


def extract(kind: str, obj, is_mask: bool = False) -> ImageInput | None:
    """Read one image into plain Python/numpy data (main thread only).

    ``is_mask`` marks an image used as a soft mask or a stencil of another
    one: it has to stay DeviceGray.
    """
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
                          can_gray="/Decode" not in obj,
                          can_index=(not is_mask and not mask and "/Decode" not in obj
                                     and "/Matte" not in obj and _base_space(colorspace)))
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
        if item.kind == "jpeg" and item.jpeg is not None:
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
    if rows is None:
        return None
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
    candidates: list[tuple] = []
    if bits != 1 or not item.ccitt_source:
        data, parms = _predicted(rows, bits, ncomp, width)
        candidates.append((data, Name.FlateDecode, parms, None))
    if bits == 1 and jbig2_ok:
        coded = _jbig2_generic(rows, width, height, scratch)
        if coded is not None:
            candidates.append((coded, Name.JBIG2Decode, None, None))
    if bits == 8 and item.can_index and ncomp in (1, 3):
        # A picture of few colours as indices into a table: a third of the
        # samples for RGB, half or less for gray with up to 16 levels.
        found = imagedata.exact_palette(
            rows[:, :width * ncomp].reshape(height, width, ncomp), PALETTE_LIMIT[ncomp])
        if found is not None:
            indices, table = found
            index_bits = imagedata.index_bits(len(table))
            data, parms = _predicted(imagedata.pack_indices(indices, index_bits),
                                     index_bits, 1, width)
            candidates.append((data, Name.FlateDecode, parms, (index_bits, table.tobytes())))
    if not candidates:
        return None
    data, filter_name, parms, palette = min(
        candidates, key=lambda c: len(c[0]) + (len(c[3][1]) if c[3] else 0))
    if len(data) + (len(palette[1]) if palette else 0) >= item.stored * MIN_GAIN:
        return None
    if filter_name == Name.JBIG2Decode:
        notes.append("jbig2")
    table = None
    if palette is not None:
        updates["/BitsPerComponent"], table = palette
        notes.append("palette")
    return Replacement(data, filter_name, parms, updates,
                       "+".join(notes) or "predictor", table)


def _apply(obj, result: Replacement) -> None:
    obj.write(result.data, filter=result.filter, decode_parms=result.parms)
    for key, value in result.updates.items():
        obj[key] = value
    if result.table is not None:
        # The table holds colours of the image's own colour space (the one
        # just set, for RGB that turned out gray), which becomes the base.
        base = obj.get("/ColorSpace")
        components = imagedata.components(base) or 1
        obj.ColorSpace = pikepdf.Array([Name.Indexed, base,
                                        len(result.table) // components - 1,
                                        pikepdf.String(result.table)])


def optimize_images(pdf: pikepdf.Pdf, scratch: Path, jobs: int, *,
                    transforms: bool, jbig2: bool,
                    progress: Callback = nothing) -> tuple[set[int], Counter]:
    """Rewrite images in place.

    Returns the pages that need a visual check and a count of rewrites by kind.
    """
    tasks: dict[tuple, list] = {}   # objgen -> [kind, obj, pages]
    masks: set[tuple] = set()       # images used as a mask of another image

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
                masks.add(child.objgen)

    visual: set[int] = set()
    counts: Counter = Counter()
    entries = list(tasks.values())
    window = max(4, jobs * 2)   # bounds the decoded pixels held in memory
    with tempfile.TemporaryDirectory(dir=scratch) as base, \
            ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for start in range(0, len(entries), window):
            batch = []
            for offset, (kind, obj, pages) in enumerate(entries[start:start + window]):
                item = extract(kind, obj, obj.objgen in masks)
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
            progress(min(1.0, (start + window) / len(entries)))
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


def ect_streams(pdf: pikepdf.Pdf, jobs: int, progress: Callback = nothing) -> bool:
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

    # Progress counts bytes: one big stream may take most of the time.
    total = sum(len(item[1]) for item in items) or 1
    lock = threading.Lock()
    done = [0]

    def pack(item):
        packed = deflate_zlib(item[1], ect_mode(len(item[1])))
        with lock:
            done[0] += len(item[1])
            fraction = done[0] / total
        progress(fraction)
        return packed

    # The compressor releases the GIL, so threads scale across cores.
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        packed = list(pool.map(pack, items))
    for (stream, _, old), new in zip(items, packed):
        if old is not None and len(new) >= old:
            continue
        stream.write(new, filter=Name.FlateDecode,
                     decode_parms=stream.get("/DecodeParms"))  # type: ignore[arg-type]
    return True


# ------------------------------------------------------------------- build --

def save(pdf: pikepdf.Pdf, target: Path, *, deflated: bool) -> None:
    """Write ``pdf`` with object and xref streams.

    PDF/A-1 is PDF 1.4 and forbids both, so a file that claims it keeps its
    classic structure (and its version) and stays valid.
    """
    mode = (pikepdf.ObjectStreamMode.preserve if pdfa_part(pdf) == 1
            else pikepdf.ObjectStreamMode.generate)
    pdf.save(target, object_stream_mode=mode,
             compress_streams=True, recompress_flate=not deflated,
             fix_metadata_version=False, deterministic_id=True)


# ------------------------------------------------- object and xref streams --
#
# qpdf builds the object streams and the cross-reference stream itself and
# packs them with zlib; together they are about a tenth of a typical file, and
# ECT makes them some 14 % smaller. They cannot be handed to qpdf already
# packed, so the saved file is rewritten: the streams get new data, every
# object moves, and the cross-reference stream is written anew.

_OBJECT = re.compile(rb"(\d+) (\d+) obj\n<<(.*?)>>\nstream\n", re.S)
_LENGTH = re.compile(rb"/Length (\d+)(?= |>|/|$)")


def _int_array(text: bytes, key: bytes) -> list[int] | None:
    found = re.search(re.escape(key) + rb" ?\[([\d\s]*)\]", text)
    return [int(v) for v in found.group(1).split()] if found else None


def _xref_rows(raw: bytes, columns: int, predictor: int | None) -> list[bytes] | None:
    """The rows of a decoded cross-reference stream (PNG predictors undone)."""
    if predictor is None:
        return [raw[i:i + columns] for i in range(0, len(raw), columns)]
    if predictor < 10:
        return None
    rows, previous = [], bytes(columns)
    for start in range(0, len(raw), columns + 1):
        kind, line = raw[start], bytearray(raw[start + 1:start + 1 + columns])
        if kind == 2:
            for i in range(columns):
                line[i] = (line[i] + previous[i]) & 255
        elif kind != 0:
            return None             # qpdf writes only "None" and "Up" rows
        rows.append(bytes(line))
        previous = bytes(line)
    return rows


def repack_xref_streams(path: Path, jobs: int = 1) -> int:
    """Pack the object streams and the cross-reference stream of a file that
    pikepdf saved with ECT; returns the bytes saved (0 when nothing changed).

    Anything unexpected leaves the file as it is, and the result is reopened
    and compared with the original before it replaces it.
    """
    if _ectdeflate is None:
        return 0
    data = path.read_bytes()
    try:
        start = int(data[data.rindex(b"startxref") + 9:].split()[0])
    except (ValueError, IndexError):
        return 0
    head = _OBJECT.match(data, start)
    if head is None or b"/Type /XRef" not in head.group(3):
        return 0                    # a classic xref table (PDF/A-1, old files)
    xref_number, xref_dict = int(head.group(1)), head.group(3)
    length = _LENGTH.search(xref_dict)
    widths = _int_array(xref_dict, b"/W")
    size = re.search(rb"/Size (\d+)", xref_dict)
    index = _int_array(xref_dict, b"/Index") or ([0, int(size.group(1))] if size else None)
    columns = sum(widths) if widths else 0
    if (length is None or widths is None or len(widths) != 3 or index is None
            or b"/Prev" in xref_dict or not columns):
        return 0
    predictor = re.search(rb"/Predictor (\d+)", xref_dict)
    try:
        raw = zlib.decompress(data[head.end():head.end() + int(length.group(1))])
    except zlib.error:
        return 0
    rows = _xref_rows(raw, columns, int(predictor.group(1)) if predictor else None)
    numbers = [n for first, count in zip(index[::2], index[1::2])
               for n in range(first, first + count)]
    if rows is None or len(rows) != len(numbers):
        return 0

    def field_value(row: bytes, number: int, default: int) -> int:
        offset = sum(widths[:number])
        part = row[offset:offset + widths[number]]
        return int.from_bytes(part, "big") if part else default

    entries = [(field_value(row, 0, 1), field_value(row, 1, 0), field_value(row, 2, 0))
               for row in rows]
    placed = sorted((entry[1], number) for number, entry in zip(numbers, entries)
                    if entry[0] == 1)
    if not placed or placed[-1] != (start, xref_number):
        return 0
    bounds = [offset for offset, _ in placed[1:]] + [len(data)]
    objects = []                    # (number, bytes) in file order, xref last
    for (offset, number), end in zip(placed, bounds):
        if not data.startswith(b"%d 0 obj" % number, offset):
            return 0
        objects.append((number, data[offset:end]))

    def repack(chunk: bytes) -> bytes:
        found = _OBJECT.match(chunk)
        if (found is None or b"/Type /ObjStm" not in found.group(3)
                or b"/DecodeParms" in found.group(3)
                or not re.search(rb"/Filter /FlateDecode\b", found.group(3))):
            return chunk
        size = _LENGTH.search(found.group(3))
        if size is None:
            return chunk
        old = chunk[found.end():found.end() + int(size.group(1))]
        try:
            plain = zlib.decompress(old)
        except zlib.error:
            return chunk
        new = deflate_zlib(plain, ect_mode(len(plain)))
        if len(new) >= len(old):
            return chunk
        header = (chunk[:found.start(3)] + _LENGTH.sub(b"/Length %d" % len(new), found.group(3),
                                                       count=1) + b">>\nstream\n")
        return header + new + chunk[found.end() + len(old):]

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        packed = list(pool.map(repack, [chunk for _, chunk in objects[:-1]]))
    out = bytearray(data[:placed[0][0]])
    moved: dict[int, int] = {}
    for (number, _), chunk in zip(objects, packed):
        moved[number] = len(out)
        out += chunk
    moved[xref_number] = len(out)
    if moved[xref_number] >= 256 ** widths[1]:
        return 0
    table = bytearray()
    previous = bytes(columns + 1)
    for number, (kind, second, third) in zip(numbers, entries):
        line = b"".join(value.to_bytes(width, "big") for value, width in zip(
            (kind, moved[number] if kind == 1 else second, third), widths))
        table += b"\x02" + bytes((a - b) & 255 for a, b in zip(line, previous[1:]))
        previous = b"\x02" + line
    stream = deflate_zlib(bytes(table), 9)
    parms = b"/DecodeParms << /Columns %d /Predictor 12 >>" % columns
    text = re.sub(rb"/DecodeParms ?<<[^>]*>>", parms, xref_dict, count=1) if predictor \
        else xref_dict + b" " + parms
    text = _LENGTH.sub(b"/Length %d" % len(stream), text, count=1)
    out += b"%d 0 obj\n<<%s>>\nstream\n" % (xref_number, text) + stream
    out += b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % moved[xref_number]
    if len(out) >= len(data):
        return 0
    candidate = path.with_name(path.name + ".repacked")
    candidate.write_bytes(bytes(out))
    try:
        if not _same_objects(path, candidate):
            return 0
        saved = len(data) - len(out)
        candidate.replace(path)
        return saved
    finally:
        candidate.unlink(missing_ok=True)


def _token(value):
    """A comparable form of a trailer entry: its object number if indirect."""
    if isinstance(value, pikepdf.Object):
        return value.objgen if value.is_indirect else value.unparse()
    return repr(value)


def _same_objects(first: Path, second: Path) -> bool:
    """Both files open without repairs and hold the same objects."""
    try:
        with pikepdf.open(first) as a, pikepdf.open(second) as b:
            if b.get_warnings() or len(a.objects) != len(b.objects):
                return False
            # The trailer is the dictionary of the xref stream: only the
            # entries that are not about the stream itself must agree.
            if any(_token(a.trailer.get(key)) != _token(b.trailer.get(key))
                   for key in ("/Root", "/Info", "/ID", "/Encrypt", "/Size")):
                return False
            for x, y in zip(a.objects, b.objects):
                if x.objgen != y.objgen or x.unparse() != y.unparse():
                    return False
                # The repacked streams differ by design; the objects they
                # hold were compared above.
                if isinstance(x, pikepdf.Stream) and x.get("/Type") not in (
                        "/ObjStm", "/XRef") and x.read_raw_bytes() != y.read_raw_bytes():
                    return False
    except (pikepdf.PdfError, OSError, ValueError):
        return False
    return True


def build(source: Path, target: Path, scratch: Path, jobs: int = 1, *,
          transforms: bool = True, jbig2: bool = True,
          ect: bool = True, cut_fonts: bool = True,
          skip_fonts: frozenset = frozenset(),
          progress: Callback = nothing) -> Report:
    """Run the whole lossless pipeline from ``source`` into ``target``.

    ``cut_fonts`` allows the font step (TrueType cutting, Type 1 to CFF);
    ``skip_fonts`` lists font programs (``objgen``) it must leave alone.
    ``progress`` gets the completed fraction (0..1) of this call.
    """
    notes: list[str] = []
    with open_pdf(source) as pdf:
        if drop_thumbnails(pdf):
            notes.append("thumbs")
        if merge_identical_streams(pdf):
            notes.append("dedupe")
        progress(0.05)
        visual, counts = optimize_images(pdf, scratch, jobs,
                                         transforms=transforms, jbig2=jbig2,
                                         progress=lambda f: progress(0.05 + 0.45 * f))
        notes.extend(f"{name}×{count}" for name, count in sorted(counts.items()))
        changed: dict[tuple, set[int]] = {}
        if transforms and cut_fonts:
            found = fonts.subset_fonts(pdf, skip_fonts)
            changed = found.programs
            visual |= found.pages
            if found.cut:
                notes.append(f"fonts×{found.cut}")
            if found.converted:
                notes.append(f"type1→cff×{found.converted}")
        progress(0.55)
        deflated = ect_streams(pdf, jobs, lambda f: progress(0.55 + 0.35 * f)) if ect else False
        progress(0.9)
        save(pdf, target, deflated=deflated)
    return Report(visual, notes, changed)
