"""Decoding image XObjects and lossless re-encoding helpers."""

from __future__ import annotations

from io import BytesIO
import struct
import zlib

import numpy as np
import pikepdf
from PIL import Image

from .pdfmodel import filters_of

GENERAL_FILTERS = {"/FlateDecode", "/LZWDecode", "/ASCII85Decode",
                   "/ASCIIHexDecode", "/RunLengthDecode"}
# Largest decoded image handled (a 600 dpi A2 colour scan is ~300 MB).
MAX_SAMPLE_BYTES = 512 << 20
_COMPONENTS = {"/DeviceGray": 1, "/DeviceRGB": 3, "/DeviceCMYK": 4,
               "/CalGray": 1, "/CalRGB": 3}


def components(colorspace) -> int | None:
    """Colour components per pixel for the colour spaces we understand."""
    if colorspace is None:
        return None
    if isinstance(colorspace, pikepdf.Name):
        return _COMPONENTS.get(str(colorspace))
    if isinstance(colorspace, pikepdf.Array) and len(colorspace):
        kind = str(colorspace[0])
        if kind == "/ICCBased":
            try:
                return int(colorspace[1].get("/N"))  # type: ignore[arg-type]
            except (TypeError, ValueError, pikepdf.PdfError):
                return None
        if kind == "/Indexed":
            return 1
        return _COMPONENTS.get(kind)
    return None


def device_family(colorspace) -> str | None:
    if isinstance(colorspace, pikepdf.Name):
        return {"/DeviceGray": "gray", "/DeviceRGB": "rgb"}.get(str(colorspace))
    return None


def jpeg_bytes(obj) -> bytes | None:
    """The JPEG stream of a DCT image (unwrapping an outer Flate), if any."""
    filters = filters_of(obj)
    try:
        if filters == ["/DCTDecode"]:
            return bytes(obj.read_raw_bytes())
        if filters == ["/FlateDecode", "/DCTDecode"]:
            parms = obj.get("/DecodeParms")
            if parms is not None and any(
                    p is not None and "/Predictor" in p
                    for p in (parms if isinstance(parms, pikepdf.Array) else [parms])):
                return None
            inflate = zlib.decompressobj()
            data = inflate.decompress(bytes(obj.read_raw_bytes()), MAX_SAMPLE_BYTES)
            return None if inflate.unconsumed_tail else data
    except (zlib.error, pikepdf.PdfError, ValueError):
        return None
    return None


def raw_samples(obj, width: int, height: int, ncomp: int,
                bits: int) -> np.ndarray | None:
    """Decoded sample rows of a generally-filtered image as (height, bytes).

    Images above MAX_SAMPLE_BYTES are skipped, and a plain Flate stream is
    inflated only up to the size the image needs, so a decompression bomb
    cannot exhaust memory.
    """
    filters = filters_of(obj)
    if any(name not in GENERAL_FILTERS for name in filters):
        return None
    row = (width * ncomp * bits + 7) // 8
    size = row * height
    if size <= 0 or size > MAX_SAMPLE_BYTES:
        return None
    try:
        if filters == ["/FlateDecode"] and "/DecodeParms" not in obj:
            data = zlib.decompressobj().decompress(bytes(obj.read_raw_bytes()), size)
        else:
            data = obj.read_bytes()
    except (pikepdf.PdfError, ValueError, zlib.error):
        return None
    if len(data) < size:
        return None
    return np.frombuffer(data, dtype=np.uint8, count=size).reshape(height, row)


def pack_bits(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.astype(np.uint8), axis=1)


def _png_candidates(rows: np.ndarray, bpp: int) -> np.ndarray:
    left = np.zeros_like(rows)
    left[:, bpp:] = rows[:, :-bpp]
    up = np.zeros_like(rows)
    up[1:] = rows[:-1]
    upper_left = np.zeros_like(rows)
    upper_left[1:, bpp:] = rows[:-1, :-bpp]
    average = ((left.astype(np.uint16) + up) // 2).astype(np.uint8)
    p = left.astype(np.int16) + up - upper_left
    pa, pb, pc = np.abs(p - left), np.abs(p - up), np.abs(p - upper_left)
    paeth = np.where((pa <= pb) & (pa <= pc), left,
                     np.where(pb <= pc, up, upper_left)).astype(np.uint8)
    return np.stack([rows, rows - left, rows - up, rows - average,
                     rows - paeth])


def png_predict(rows: np.ndarray, bpp: int) -> bytes:
    """PNG-filter every row, choosing the filter with the smallest residual
    sum (the classic libpng heuristic)."""
    height = rows.shape[0]
    candidates = _png_candidates(rows, max(1, bpp))
    cost = np.abs(candidates.view(np.int8).astype(np.int16)).sum(axis=2)
    choice = cost.argmin(axis=0)
    out = np.empty((height, rows.shape[1] + 1), dtype=np.uint8)
    out[:, 0] = choice
    out[:, 1:] = candidates[choice, np.arange(height)]
    return out.tobytes()


def g4_bitmap(data: bytes, width: int, height: int, black_is_1: bool) -> np.ndarray | None:
    """Decode a Group 4 stream through a TIFF wrapper -> (height, bytes) rows
    holding the sample bits a PDF reader would see.

    A PDF Group 4 stream codes black runs as black ink, which is TIFF's
    WhiteIsZero convention (photometric 0).
    """
    entries = [(256, 4, width), (257, 4, height), (258, 3, 1), (259, 3, 4),
               (262, 3, 0), (273, 4, 0), (278, 4, height), (279, 4, len(data))]
    offset = 8 + 2 + 12 * len(entries) + 4
    ifd = struct.pack("<H", len(entries))
    for tag, kind, value in entries:
        if tag == 273:
            value = offset
        ifd += struct.pack("<HHI", tag, kind, 1) + (
            struct.pack("<HH", value, 0) if kind == 3 else struct.pack("<I", value))
    blob = b"II*\x00" + struct.pack("<I", 8) + ifd + b"\x00\x00\x00\x00" + data
    try:
        with Image.open(BytesIO(blob)) as tiff:
            if tiff.size != (width, height):
                return None
            bitmap = tiff.convert("1")
            packed = np.frombuffer(bitmap.tobytes(), dtype=np.uint8).reshape(
                height, (width + 7) // 8).copy()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None
    # Pillow reports black as 0; with BlackIs1 a PDF reader sees 1.
    return ~packed if black_is_1 else packed


def to_png(rows: np.ndarray, width: int, height: int) -> bytes:
    """1-bit packed rows (0 = black) as PNG for jbig2enc."""
    bitmap = Image.frombytes("1", (width, height), np.ascontiguousarray(rows).tobytes())
    buffer = BytesIO()
    bitmap.save(buffer, "PNG")
    return buffer.getvalue()


def box_blur(plane: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, clipped at the edges (float32)."""
    out = plane.astype(np.float32, copy=False)
    for axis in (0, 1):
        size = out.shape[axis]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius + 1, radius)
        total = np.cumsum(np.pad(out, pad, mode="constant"), axis=axis,
                          dtype=np.float64)
        upper = np.take(total, np.arange(2 * radius + 1, 2 * radius + 1 + size), axis=axis)
        lower = np.take(total, np.arange(0, size), axis=axis)
        out = (upper - lower).astype(np.float32)
    return out


def bitmap_from(jpeg: bytes | None, rows: np.ndarray | None, width: int,
                height: int, ncomp: int) -> Image.Image | None:
    """PIL image (RGB or L) from a JPEG stream or decoded 8-bit rows."""
    try:
        if jpeg is not None:
            with Image.open(BytesIO(jpeg)) as decoded:
                if decoded.mode not in ("RGB", "L"):
                    return None
                return decoded.copy()
        if rows is None or ncomp not in (1, 3):
            return None
        shape = (height, width) if ncomp == 1 else (height, width, 3)
        return Image.fromarray(rows.reshape(shape), "L" if ncomp == 1 else "RGB")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


# ---------------------------------------------------------------- palettes --

def pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    """(height, width) indices -> rows of ``bits``-bit samples, as PDF stores them."""
    if bits == 8:
        return np.ascontiguousarray(indices, dtype=np.uint8)
    per = 8 // bits
    height, width = indices.shape
    padded = np.zeros((height, -(-width // per) * per), dtype=np.uint8)
    padded[:, :width] = indices
    packed = np.zeros((height, padded.shape[1] // per), dtype=np.uint8)
    for position in range(per):
        packed |= (padded[:, position::per] << (8 - bits * (position + 1))).astype(np.uint8)
    return packed


def index_bits(count: int) -> int:
    return 1 if count <= 2 else 2 if count <= 4 else 4 if count <= 16 else 8


def exact_palette(pixels: np.ndarray, limit: int = 256):
    """(indices, table) when an image (height, width, components) has at most
    ``limit`` different colours, else None. Nothing is lost."""
    height, width, components = pixels.shape
    # A thinned sample settles most images without sorting every pixel.
    probe = pixels[::max(1, height // 64), ::max(1, width // 64)].reshape(-1, components)
    if len(np.unique(probe, axis=0)) > limit:
        return None
    key = np.zeros((height, width), dtype=np.uint32)
    for channel in range(components):
        key = key << 8 | pixels[:, :, channel]
    values, inverse = np.unique(key, return_inverse=True)
    if len(values) > limit:
        return None
    table = np.stack([(values >> (8 * (components - 1 - channel))) & 255
                      for channel in range(components)], axis=1).astype(np.uint8)
    return inverse.reshape(height, width).astype(np.uint8), table


def write_indexed(obj, data: bytes, parms: dict | None, bits: int, table: bytes,
                  count: int) -> None:
    base = obj.get("/ColorSpace")
    obj.write(data, filter=pikepdf.Name.FlateDecode,
              decode_parms=pikepdf.Dictionary(**parms) if parms else None)
    obj.ColorSpace = pikepdf.Array([pikepdf.Name.Indexed, base, count - 1, pikepdf.String(table)])
    obj.BitsPerComponent = bits
