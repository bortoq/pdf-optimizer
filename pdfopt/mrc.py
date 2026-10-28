"""Layered (MRC) compression of scanned pages.

A page that is one big picture becomes three layers:

* background — the page with the text removed, at low resolution
  (JPEG or JPEG 2000, searched against an SSIM budget);
* ink — a full-resolution 1-bit stencil (JBIG2 with shared symbol
  dictionaries, Group 4 or Flate, whichever is smallest);
* foreground colour — one colour per page; a stencil per colour when the
  text uses a few distinct inks (like DjVu's per-glyph colours); or a tiny
  image painted through the stencil when the colours do not form a palette.

Every page is rendered and compared with the source afterwards, and pages
that do not pass are restored.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import shutil
import zlib

import numpy as np
import pikepdf
from pikepdf import Name
from PIL import Image, ImageFilter, TiffImagePlugin

from . import imagedata, jpegopt, lossless
from .pdfmodel import MRC_FORM_KEY, filters_of, open_pdf, page_resources
from .render import SSIM_DPI, VisualGate
from .ssim import ssim
from .tools import run, which

BACKGROUND_DPIS = (100, 50)     # tried in this order, then 200 as a rescue
MAX_INK = 0.30
MIN_SIDE = 500
MIN_RAW_BYTES = 50_000
JBIG2_CHUNK = 64
# Symbol-match threshold by glyph size. Small glyphs in lossy symbol mode
# swap look-alikes (the Xerox 6/8 case): on 5 pt text at 300 dpi -t 0.80
# turned 62 sixes into eights, on 4 pt even -t 0.85 swapped 8/6 and e/a.
# Size is the 90th percentile of vertical ink runs (stem height, px), per
# band of text rows: (minimum stem, threshold), else JBIG2_LOSSLESS. 5 pt
# digits measure 13 px and still swapped at 0.80, hence the margin; 10 pt
# text at 300 dpi measures 18-19 px. Below 10 px no threshold is safe (on
# 3.5 pt glyphs even -t 0.92 turned 9 into 8), so such text is coded
# losslessly. jbig2enc 0.28 (Ubuntu 24.04) rejects thresholds above 0.9.
JBIG2_THRESHOLDS = ((15, 0.80), (12, 0.85), (10, 0.90))
JBIG2_LOSSLESS = 1.0             # generic region coding, no symbol matching
PREVIEW_MARGIN = 0.012
# Compression ratios, strongest first: a flat paper background is tiny in
# JPEG 2000 (~1 KB per page at 100 dpi) where JPEG cannot go below ~4 KB.
JPX_RATES = (2000, 1000, 600, 400, 250, 150, 100, 50)
# Ink palette: colours are measured on cells of 1/INK_CELL_DPI inch. Distances
# weigh lightness by PALETTE_LIGHTNESS, since thin strokes of one ink look
# lighter than thick ones; colours closer than PALETTE_MERGE are one ink, and
# a page whose cells sit further than PALETTE_FIT from their ink (90th
# percentile) keeps the colour image instead.
INK_CELL_DPI = 40
PALETTE_LIGHTNESS = 0.35
PALETTE_MERGE = 48.0
PALETTE_FIT = 40.0
PALETTE_SUPPORT = 20            # cells an ink needs to become a palette entry
MAX_INKS = 6
# A page may use the lower MRC threshold only when the loss comes from
# bilevel text edges: every inked tile of the ideal-background composite must
# keep this SSIM (a posterized photo drops to ~0.6, dense text stays >0.9).
TEXT_TILE = 128
TEXT_TILE_FLOOR = 0.85


@dataclass
class PageJob:
    number: int                      # 1-based page number
    jpeg: bytes | None
    rows: np.ndarray | None
    width: int
    height: int
    ncomp: int
    stored: int
    page_size: tuple[float, float]


@dataclass
class Scan:
    """A scanned page image that may be layered (metadata only)."""

    image: pikepdf.Object
    users: list                      # (page number, XObject table, name)
    size: tuple[float, float]
    width: int
    height: int
    ncomp: int


@dataclass
class Ink:
    color: tuple[float, float, float]
    mask: bytes                      # Flate or Group 4 encoded stencil
    group4: bool
    stencil_png: bytes
    threshold: float                 # JBIG2 symbol-match threshold it allows


@dataclass
class Foreground:
    """How the ink is coloured: stencils with their colours, and for previews
    the colour of every ink cell (None for a single colour)."""

    inks: list[tuple[tuple[float, float, float], np.ndarray]]
    image: tuple[bytes, tuple[int, int]] | None = None   # JPEG through inks[0]
    tint: Image.Image | None = None


@dataclass
class Layers:
    background: bytes
    bg_size: tuple[int, int]
    codec: str                       # "/DCTDecode" or "/JPXDecode"
    grayscale: bool
    inks: list[Ink]
    mask_size: tuple[int, int]
    foreground: tuple[bytes, tuple[int, int]] | None
    minimum: float                   # SSIM the page has to keep


# --------------------------------------------------------------- segmenting --

def ink_mask(gray: np.ndarray, dpi: float) -> np.ndarray:
    """Boolean ink mask; copes with faded scans and light-on-dark text."""
    height, width = gray.shape
    factor = max(1, int(dpi // 60))
    rows, cols = height // factor * factor, width // factor * factor
    small = gray[:rows, :cols].reshape(rows // factor, factor,
                                       cols // factor, factor).mean(axis=(1, 3))
    small = small.astype(np.uint8)
    if np.median(small) < 110:
        gray = 255 - gray
        small = 255 - small
    paper = Image.fromarray(small).filter(ImageFilter.MaxFilter(5))
    paper = imagedata.box_blur(np.asarray(paper), 3)
    paper = Image.fromarray(paper.clip(0, 255).astype(np.uint8)).resize(
        (width, height), Image.Resampling.BILINEAR)
    paper = np.asarray(paper, dtype=np.int16)
    contrast = paper - gray.astype(np.int16)
    ink = contrast > np.maximum(30, (paper * 3) // 10)
    if dpi >= 250:
        # 3x3 majority vote: drops specks and smooths ragged scan edges, so
        # more glyphs match in the JBIG2 symbol dictionary.
        padded = np.pad(ink, 1)
        votes = np.zeros(ink.shape, dtype=np.uint8)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                votes += padded[dy:dy + height, dx:dx + width]
        ink = votes >= 5
    return ink


def stem_height(ink: np.ndarray) -> float:
    """90th percentile of vertical ink run lengths: the glyph size in pixels."""
    edges = np.diff(np.pad(ink, ((1, 1), (0, 0))).astype(np.int8), axis=0).T.ravel()
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(np.percentile(runs, 90)) if len(runs) else 0.0


def symbol_threshold(stem: float) -> float:
    """JBIG2 symbol-match threshold that is safe for glyphs of ``stem`` px,
    or JBIG2_LOSSLESS when no symbol matching is."""
    for minimum, threshold in JBIG2_THRESHOLDS:
        if stem >= minimum:
            return threshold
    return JBIG2_LOSSLESS


def size_classes(ink: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Split a stencil by the threshold its text allows -> [(threshold, part)].

    Bands of rows between blank rows (text lines, or runs of lines) are sized
    separately, so small print below normal text (footnotes, captions) gets
    its own, stricter threshold instead of the page's.
    """
    rows = ink.any(axis=1)
    edges = np.diff(np.concatenate(([0], rows.astype(np.int8), [0])))
    band = np.zeros(len(rows))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        band[start:stop] = symbol_threshold(stem_height(ink[start:stop]))
    return [(threshold, ink & (band == threshold)[:, None])
            for threshold in sorted(set(band[rows].tolist()))]


def _fill_background(small: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Replace unknown pixels by a normalized average of nearby known ones."""
    filled = small.copy()
    known = known.copy()
    for radius in (2, 4, 8, 16, 32):
        if known.all():
            break
        weight = known.astype(np.float32)
        den = imagedata.box_blur(weight, radius)
        usable = ~known & (den > 0.02)
        for channel in range(filled.shape[2]):
            num = imagedata.box_blur(filled[:, :, channel] * weight, radius)
            filled[:, :, channel][usable] = num[usable] / den[usable]
        known |= usable
    if not known.all():
        for channel in range(filled.shape[2]):
            plane = filled[:, :, channel]
            plane[~known] = np.median(plane[known]) if known.any() else 255
    return filled


def background_image(bitmap: Image.Image, ink: Image.Image,
                     size: tuple[int, int]) -> Image.Image:
    """Page at ``size`` with the ink pixels filled in from their surroundings."""
    small = np.asarray(bitmap.resize(size, Image.Resampling.BOX), dtype=np.float32)
    if small.ndim == 2:
        small = small[:, :, None]
    coverage = ink.resize(size, Image.Resampling.BOX).filter(ImageFilter.MaxFilter(5))
    known = np.asarray(coverage) == 0
    result = _fill_background(small, known)
    result = result.clip(0, 255).astype(np.uint8)
    return Image.fromarray(result[:, :, 0] if bitmap.mode == "L" else result,
                           bitmap.mode)


def _ink_space(colours: np.ndarray) -> np.ndarray:
    """RGB -> (damped lightness, chroma), the space inks are told apart in."""
    lightness = colours.mean(axis=-1, keepdims=True)
    return np.concatenate([lightness * PALETTE_LIGHTNESS, colours - lightness], axis=-1)


def _cell_colours(rgb: np.ndarray, ink: np.ndarray, cell: int):
    """Mean colour of the darker half of the ink in every cell, ink counts."""
    height, width = ink.shape
    rows, cols = -(-height // cell), -(-width // cell)
    pad = ((0, rows * cell - height), (0, cols * cell - width))

    def cells(plane):
        return np.pad(plane, pad).reshape(rows, cell, cols, cell).sum(axis=(1, 3))

    lightness = rgb.mean(axis=2, dtype=np.float32)
    count = cells(ink.astype(np.float32))
    mean = cells(lightness * ink) / np.maximum(count, 1)
    # Anti-aliased rims only dilute the colour; the darker half carries it.
    core = ink & (lightness <= np.repeat(np.repeat(mean, cell, 0), cell, 1)[:height, :width])
    weight = cells(core.astype(np.float32))
    colours = np.stack([cells(rgb[:, :, c] * core) for c in range(3)], axis=2)
    return colours / np.maximum(weight, 1)[:, :, None], count


def _palette(colours: np.ndarray, known: np.ndarray):
    """Inks of the known cells -> (palette RGB, distance of each known cell).

    Farthest-point seeding (a new ink needs PALETTE_SUPPORT cells beyond
    PALETTE_MERGE of the others) followed by a few k-means rounds.
    """
    sample = colours[known]
    points = _ink_space(sample)
    centres = [np.median(points, axis=0)]
    while len(centres) < MAX_INKS:
        nearest = np.sqrt(((points[:, None] - np.array(centres)[None]) ** 2)
                          .sum(axis=2)).min(axis=1)
        far = points[nearest > PALETTE_MERGE]
        if len(far) < PALETTE_SUPPORT:
            break
        # Seed at the densest far colour, not at a stray outlier.
        probe = far[:2000]
        density = (((far[:, None] - probe[None]) ** 2).sum(axis=2)
                   < (PALETTE_MERGE / 2) ** 2).sum(axis=1)
        centres.append(far[np.argmax(density)])
    centres = np.array(centres)
    for _ in range(8):
        label = ((points[:, None] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)
        kept = [k for k in range(len(centres)) if (label == k).sum() >= PALETTE_SUPPORT]
        centres = np.array([points[label == k].mean(axis=0) for k in kept]) \
            if kept else np.median(points, axis=0)[None]
    label = ((points[:, None] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)
    # The last update may leave a centre without cells; it has no colour.
    centres = centres[np.isin(np.arange(len(centres)), label)]
    label = ((points[:, None] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)
    palette = np.array([sample[label == k].mean(axis=0) for k in range(len(centres))])
    return palette, np.sqrt(((points - centres[label]) ** 2).sum(axis=1))


def _palette_inks(rgb: np.ndarray, ink: np.ndarray, dpi: float) -> Foreground | None:
    """One stencil per ink colour, or None unless the ink forms a palette of
    two or more colours."""
    cell = max(2, round(dpi / INK_CELL_DPI))
    colours, count = _cell_colours(rgb, ink, cell)
    known = count >= 4
    if known.sum() < PALETTE_SUPPORT:
        return None
    palette, distance = _palette(colours, known)
    if np.percentile(distance, 90) > PALETTE_FIT:
        return None
    inked = count > 0
    space = _ink_space(colours[inked])
    label = np.zeros(count.shape, dtype=np.intp)
    label[inked] = ((space[:, None] - _ink_space(palette)[None]) ** 2).sum(axis=2).argmin(axis=1)
    # Majority over 3x3 cells (weighted by ink) removes stray single cells.
    votes = np.stack([imagedata.box_blur(count * (label == k), 1)
                      for k in range(len(palette))])
    label = np.where(inked, votes.argmax(axis=0), 0)
    pixels = np.repeat(np.repeat(label, cell, 0), cell, 1)[:ink.shape[0], :ink.shape[1]]
    inks = [(tuple(float(v) / 255 for v in palette[k]), ink & (pixels == k))
            for k in range(len(palette))]
    inks = [(color, stencil) for color, stencil in inks if stencil.any()]
    if len(inks) < 2:
        # One ink whose cores still vary (colour noise, faint strokes): the
        # colour image follows those variations better than a flat colour.
        return None
    tint = Image.fromarray(palette[label].clip(0, 255).astype(np.uint8), "RGB")
    return Foreground(inks, tint=tint)


def _foreground(rgb: np.ndarray, ink: np.ndarray, dpi: float) -> Foreground:
    """Ink colour: one colour, one stencil per ink, or a small colour image."""
    positions = np.flatnonzero(ink.ravel())
    if len(positions) == 0:
        return Foreground([((0.0, 0.0, 0.0), ink)])
    step = max(1, len(positions) // 200_000)
    sample = rgb.reshape(-1, 3)[positions[::step]].astype(np.float32)
    # Anti-aliased edges vary in brightness, so only the dark cores say
    # whether the text really uses several colours.
    brightness = sample.mean(axis=1)
    cores = sample[brightness > 145] if np.median(brightness) > 128 \
        else sample[brightness < 110]
    cores = cores if len(cores) >= 50 else sample
    color = tuple(float(v) / 255 for v in np.median(cores, axis=0))
    if cores.std(axis=0).max() < 28:
        return Foreground([(color, ink)])
    found = _palette_inks(rgb, ink, dpi)
    if found is not None:
        return found
    factor = max(8, round(dpi / 25))
    grid = (max(1, rgb.shape[1] // factor), max(1, rgb.shape[0] // factor))
    weight = np.asarray(Image.fromarray(ink.astype(np.uint8) * 255).resize(
        grid, Image.Resampling.BOX), dtype=np.float32) / 255
    planes = []
    for channel in range(3):
        part = Image.fromarray((rgb[:, :, channel] * ink).astype(np.float32), "F")
        planes.append(np.asarray(part.resize(grid, Image.Resampling.BOX)))
    mean = np.stack(planes, axis=2) / np.maximum(weight, 1e-4)[:, :, None]
    mean[weight < 1e-3] = np.array(color) * 255
    image = Image.fromarray(mean.clip(0, 255).astype(np.uint8), "RGB")
    data = jpegopt.tighten(jpegopt.encode(image, 60))
    with Image.open(BytesIO(data)) as decoded:
        tint = decoded.convert("RGB")
    return Foreground([(color, ink)], image=(data, grid), tint=tint)


def _encode_stencil(packed: np.ndarray, width: int, height: int) -> tuple[bytes, bool]:
    flate = zlib.compress(packed.tobytes(), 9)
    try:
        bitmap = Image.frombytes("1", (width, height), packed.tobytes())
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[278] = height
        output = BytesIO()
        bitmap.save(output, "TIFF", compression="group4", tiffinfo=tags)
        output.seek(0)
        with Image.open(output) as tiff:
            offsets, counts = tiff.tag_v2[273], tiff.tag_v2[279]
        if len(offsets) == len(counts) == 1:
            group4 = output.getvalue()[offsets[0]:offsets[0] + counts[0]]
            if len(group4) < len(flate):
                return group4, True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return flate, False


# ----------------------------------------------------------- layer search --

def _jpx(bitmap: Image.Image, rate: int, scratch: Path) -> bytes | None:
    opj = which("opj_compress")
    if opj is None:
        return None
    source = scratch / ("jpx.pgm" if bitmap.mode == "L" else "jpx.ppm")
    target = scratch / "jpx.jp2"
    bitmap.save(source)
    target.unlink(missing_ok=True)
    done = run([opj, "-i", str(source), "-o", str(target), "-I", "-r", str(rate),
                "-threads", "1"])
    return target.read_bytes() if done.returncode == 0 and target.is_file() else None


class _Page:
    """Per-page state shared by the background search steps."""

    def __init__(self, bitmap: Image.Image, ink: np.ndarray, fg: Foreground,
                 dpi: float, minimum: float):
        self.bitmap = bitmap
        self.ink = Image.fromarray(ink.astype(np.uint8) * 255)
        self.minimum = minimum
        width, height = bitmap.size
        factor = jpegopt.view_factor(width, height, dpi)
        self.view = (max(1, round(width * factor)), max(1, round(height * factor)))
        self.reference = jpegopt.view_of(bitmap, self.view)
        self.coverage = (np.asarray(self.ink.resize(self.view, Image.Resampling.BOX),
                                    dtype=np.float32) / 255)[:, :, None]
        self.keep = 1 - self.coverage
        self._paint: dict[int, np.ndarray] = {}
        if fg.tint is not None:
            tint = fg.tint.resize(self.view, Image.Resampling.NEAREST)
            self.tint = np.asarray(tint, dtype=np.float32)
        else:
            self.tint = np.array(fg.inks[0][0], dtype=np.float32) * 255

    def composite(self, background: Image.Image) -> np.ndarray:
        base = np.asarray(background.resize(self.view, Image.Resampling.NEAREST),
                          dtype=np.float32)
        if base.ndim == 2:
            base = base[:, :, None]
        channels = base.shape[2]
        if channels not in self._paint:   # the same for every probe of this page
            self._paint[channels] = self._tint(channels) * self.coverage
        base *= self.keep
        base += self._paint[channels]
        out = base.clip(0, 255).astype(np.uint8)
        return out[:, :, 0] if out.shape[2] == 1 else out

    def score(self, decoded: Image.Image) -> float:
        return ssim(self.reference, self.composite(decoded))

    def text_like(self) -> bool:
        """True when stencilled ink costs SSIM only at text edges.

        The page itself serves as an ideal background; every tile that holds
        ink must then keep TEXT_TILE_FLOOR.
        """
        out = self.composite(self.bitmap)
        coverage = self.coverage[:, :, 0]
        height, width = coverage.shape
        for y in range(0, height - TEXT_TILE + 1, TEXT_TILE):
            for x in range(0, width - TEXT_TILE + 1, TEXT_TILE):
                window = (slice(y, y + TEXT_TILE), slice(x, x + TEXT_TILE))
                if (coverage[window].mean() > 0.005 and
                        ssim(self.reference[window], out[window]) < TEXT_TILE_FLOOR):
                    return False
        return True

    def _tint(self, channels: int):
        if channels == 1 and isinstance(self.tint, np.ndarray) and self.tint.ndim == 3:
            return self.tint.mean(axis=2, keepdims=True)
        if channels == 1:
            return np.float32(self.tint.mean())
        return self.tint


def _best_jpeg(page: _Page, background: Image.Image):
    """Bisect the JPEG quality for one background -> (quality, bytes)|None."""
    target = min(1.0, page.minimum + PREVIEW_MARGIN)

    def passes(quality: int) -> bytes | None:
        data = jpegopt.encode(background, quality)
        with Image.open(BytesIO(data)) as decoded:
            decoded.load()
            return data if page.score(decoded) >= target else None

    return jpegopt.bisect_quality(passes)


def _background_search(page: _Page, bitmap: Image.Image, dpi: float,
                       scratch: Path):
    """Cheapest passing background -> (bytes, size, codec) or None.

    Starts at 100 dpi; a pass is followed by a try at 50 dpi, a failure by a
    try at the render resolution. The smaller of the passing results wins.
    """
    floor = min(1.0, page.minimum + PREVIEW_MARGIN)
    width, height = bitmap.size
    best = None

    def attempt(target_dpi: float):
        scale = min(1.0, target_dpi / dpi)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        background = background_image(bitmap, page.ink, size)
        found = _best_jpeg(page, background)
        if found is None:
            return None
        candidates = [(jpegopt.tighten(found[1]), "/DCTDecode")]
        for rate in JPX_RATES:
            data = _jpx(background, rate, scratch)
            if data is None:
                break
            with Image.open(BytesIO(data)) as decoded:
                decoded.load()
                if page.score(decoded) >= floor:
                    candidates.append((data, "/JPXDecode"))
                    break
        data, codec = min(candidates, key=lambda entry: len(entry[0]))
        return data, size, codec

    first = attempt(min(dpi, BACKGROUND_DPIS[0]))
    if first is not None:
        best = first
        lower = BACKGROUND_DPIS[1]
        if dpi > lower * 1.1:
            second = attempt(lower)
            if second is not None and len(second[0]) < len(best[0]):
                best = second
    elif dpi > SSIM_DPI * 1.05:
        best = attempt(SSIM_DPI)
    return best


def find_layers(job: PageJob, minimum: float, scratch: Path,
                relaxed: float | None = None) -> Layers | None:
    """Segment one page and search the cheapest background that passes.

    ``relaxed`` is the lower SSIM allowed for text pages (see ``text_like``):
    bilevel ink cannot reproduce soft scanned edges, so such pages would
    otherwise never be layered at high thresholds.
    """
    bitmap = imagedata.bitmap_from(job.jpeg, job.rows, job.width, job.height, job.ncomp)
    if bitmap is None:
        return None
    width, height = bitmap.size
    dpi = max(width * 72 / max(job.page_size[0], 1.0),
              height * 72 / max(job.page_size[1], 1.0))
    grayscale = bitmap.mode == "L"
    rgb = np.asarray(bitmap.convert("RGB"))
    gray = np.asarray(bitmap if grayscale else bitmap.convert("L"))
    ink = ink_mask(gray, dpi)
    if ink.mean() > MAX_INK:
        return None
    fg = _foreground(rgb, ink, dpi)
    del rgb
    inks = []
    for color, stencil in fg.inks:
        parts = size_classes(stencil) or [(JBIG2_LOSSLESS, stencil)]
        if fg.image is not None:
            # The colour image is painted through one stencil: keep it whole,
            # at the strictest threshold of its parts.
            parts = [(max(threshold for threshold, _ in parts), stencil)]
        for threshold, part in parts:
            packed = np.packbits(~part, axis=1)
            mask, group4 = _encode_stencil(packed, width, height)
            inks.append(Ink(color, mask, group4, imagedata.to_png(packed, width, height),
                            threshold))
    page = _Page(bitmap, ink, fg, dpi, minimum)
    if relaxed is not None and relaxed < minimum and page.text_like():
        page.minimum = relaxed

    best = _background_search(page, bitmap, dpi, scratch)
    budget = job.stored - sum(len(item.mask) + 600 for item in inks)
    if best is None or len(best[0]) >= budget:
        return None
    return Layers(best[0], best[1], best[2], grayscale, inks, (width, height),
                  fg.image, page.minimum)


def _layer_job(args):
    job, minimum, relaxed, scratch = args
    if job is None:
        return None
    directory = scratch / f"mrc-{job.number}"
    directory.mkdir(exist_ok=True)
    try:
        return find_layers(job, minimum, directory, relaxed)
    except Exception:   # one odd page must not cost the other pages their layers
        return None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------- JBIG2 ----

def jbig2_streams(stencils: list[bytes], scratch: Path, *, symbols: bool = True,
                  threshold: float = JBIG2_THRESHOLDS[-1][1]):
    """Encode stencils with jbig2enc, in chunks.

    Symbol mode (``-s``) shares one dictionary per chunk of pages and gives the
    smallest files, but may merge look-alike glyphs; ``threshold`` (see
    ``symbol_threshold``) bounds that risk. ``symbols=False`` or a
    JBIG2_LOSSLESS threshold codes every page losslessly with the generic
    region coder.

    Returns ``[(globals or None, [page streams])]`` per chunk, with None for
    chunks where jbig2enc failed; None when jbig2enc is missing.
    """
    jbig2 = which("jbig2")
    if jbig2 is None or not stencils:
        return None

    def encode(index: int):
        chunk = stencils[index:index + JBIG2_CHUNK]
        directory = scratch / f"jbig2-{index}"
        directory.mkdir()
        try:
            paths = []
            for number, data in enumerate(chunk):
                path = directory / f"s{number:05d}.png"
                path.write_bytes(data)
                paths.append(path)
            if not symbols or threshold >= JBIG2_LOSSLESS:
                pages = []
                for path in paths:
                    target = path.with_suffix(".jb2")
                    done = run([jbig2, "-p", str(path)], stdout_to=target)
                    if done.returncode != 0 or not target.stat().st_size:
                        return None
                    pages.append(target.read_bytes())
                return None, pages
            base = directory / "out"
            done = run([jbig2, "-s", "-p", "-t", str(threshold), "-b", str(base),
                        *map(str, paths)])
            table = directory / "out.sym"
            pages = sorted(directory.glob("out.[0-9]*"))
            if done.returncode != 0 or not table.is_file() or len(pages) != len(chunk):
                return None
            return table.read_bytes(), [page.read_bytes() for page in pages]
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    with ThreadPoolExecutor(max_workers=min(4, len(stencils) // JBIG2_CHUNK + 1)) as pool:
        return list(pool.map(encode, range(0, len(stencils), JBIG2_CHUNK)))


# ----------------------------------------------------------- PDF assembly --

def _stream(pdf, data: bytes, **entries):
    filter_name = entries.pop("filter", None)
    parms = entries.pop("parms", None)
    stream = pikepdf.Stream(pdf, b"")
    stream.write(data, filter=filter_name, decode_parms=parms) if filter_name else \
        stream.write(data)
    for key, value in entries.items():
        stream[f"/{key}"] = value
    return pdf.make_indirect(stream)


def _image(pdf, data, size, *, filter_name, colorspace=None, mask=False, parms=None):
    return _stream(pdf, data, filter=filter_name, parms=parms, Type=Name.XObject,
                   Subtype=Name.Image, Width=size[0], Height=size[1],
                   BitsPerComponent=1 if mask else 8,
                   **({"ImageMask": True} if mask else {"ColorSpace": colorspace}))


def _fill_colour(color: tuple[float, float, float], calibrated: bool,
                 grayscale: bool) -> str:
    """Content operators that set the ink colour: DeviceRGB, or the scan's
    own colour space (/CS0) with one component for gray scans."""
    if not calibrated:
        r, g, b = color
        return f"{r:.5f} {g:.5f} {b:.5f} rg"
    values = (sum(color) / 3,) if grayscale else color
    return "/CS0 cs " + " ".join(f"{v:.5f}" for v in values) + " sc"


@dataclass
class Record:
    page: int                        # 1-based
    xobjects: pikepdf.Object
    name: str
    original: pikepdf.Object
    minimum: float                   # SSIM the layered page has to keep


def restore(record: Record) -> None:
    record.xobjects[record.name] = record.original


def scan_candidates(pdf: pikepdf.Pdf) -> list[Scan]:
    """Single-image pages that qualify for layering, one entry per image.

    ``users`` lists every page that shows the image, so identical pages
    (merged by the lossless stage) are layered once and shared.
    """
    found: dict[tuple, list] = {}
    for number, page in enumerate(pdf.pages, start=1):
        resources = page_resources(page.obj)
        xobjects = resources.get("/XObject") if resources is not None else None
        if xobjects is None:
            continue
        images = [(name, xobjects[name]) for name in xobjects
                  if xobjects[name].get("/Subtype") == "/Image"]
        if len(images) != 1:
            continue
        name, image = images[0]
        entry = found.setdefault(image.objgen, [image, []])
        entry[1].append((number, xobjects, name))
    scans = []
    for image, users in found.values():
        if (any(key in image for key in ("/Mask", "/SMask", "/Decode", "/ImageMask"))
                or image.get("/BitsPerComponent") != 8):
            continue
        width, height = int(image.get("/Width", 0)), int(image.get("/Height", 0))
        cs = image.get("/ColorSpace")
        ncomp = imagedata.components(cs)
        if (min(width, height) < MIN_SIDE or ncomp not in (1, 3) or
                (isinstance(cs, pikepdf.Array) and str(cs[0]) == "/Indexed")):
            continue
        box = [float(v) for v in pdf.pages[users[0][0] - 1].mediabox]
        scans.append(Scan(image, users, (abs(box[2] - box[0]), abs(box[3] - box[1])),
                          width, height, ncomp))
    return scans


def load_job(scan: Scan) -> PageJob | None:
    """Read the pixels of a scan (main thread; done lazily to bound memory)."""
    image, number = scan.image, scan.users[0][0]
    try:
        stored = len(image.read_raw_bytes())
        if filters_of(image) in (["/DCTDecode"], ["/FlateDecode", "/DCTDecode"]):
            data = imagedata.jpeg_bytes(image)
            if data is None:
                return None
            return PageJob(number, data, None, scan.width, scan.height,
                           scan.ncomp, stored, scan.size)
        if stored < MIN_RAW_BYTES:
            return None
        rows = imagedata.raw_samples(image, scan.width, scan.height, scan.ncomp, 8)
        if rows is None:
            return None
        return PageJob(number, None, rows, scan.width, scan.height, scan.ncomp,
                       stored, scan.size)
    except (OSError, ValueError, pikepdf.PdfError):
        return None


def _bounded_map(pool, function, items, window: int):
    """Like ``pool.map`` but never holds more than ``window`` jobs in memory."""
    pending = []
    iterator = iter(items)
    done = False
    while pending or not done:
        while not done and len(pending) < window:
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                done = True
        if pending:
            yield pending.pop(0).result()


def add_layers(pdf: pikepdf.Pdf, scratch: Path, minimum: float, jobs: int,
               log=None, *, symbols: bool = True,
               relaxed: float | None = None) -> list[Record]:
    """Replace scanned-page images by layered forms; returns undo records."""
    log = log or (lambda message: None)
    candidates = scan_candidates(pdf)
    if not candidates:
        return []
    results = []
    with ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else _Inline() as pool:
        stream = _bounded_map(
            pool, _layer_job,
            ((load_job(scan), minimum, relaxed, scratch) for scan in candidates),
            jobs * 2)
        for index, (scan, layers) in enumerate(zip(candidates, stream), start=1):
            if layers is not None:
                results.append((scan, layers))
            if len(candidates) >= 50 and index % 50 == 0:
                log(f"MRC: prepared {index}/{len(candidates)} pages")
    if not results:
        return []

    # Every ink stencil of every page, in order. Stencils that allow the same
    # symbol threshold are coded together, in chunks with shared dictionaries.
    stencils = [ink for _, layers in results for ink in layers.inks]
    symbol_streams: dict[int, tuple] = {}
    for threshold in sorted({ink.threshold for ink in stencils}):
        group = [position for position, ink in enumerate(stencils)
                 if ink.threshold == threshold]
        chunks = jbig2_streams([stencils[p].stencil_png for p in group], scratch,
                               symbols=symbols, threshold=threshold)
        for chunk_index, chunk in enumerate(chunks or []):
            if chunk is None:
                continue
            members = group[chunk_index * JBIG2_CHUNK:(chunk_index + 1) * JBIG2_CHUNK]
            jbig2_total = len(chunk[0] or b"") + sum(len(p) for p in chunk[1])
            if jbig2_total < sum(len(stencils[p].mask) for p in members):
                shared = _stream(pdf, chunk[0]) if chunk[0] is not None else None
                for position, data in zip(members, chunk[1]):
                    symbol_streams[position] = (data, shared)

    records = []
    position = 0
    for scan, layers in results:
        image, users = scan.image, scan.users
        # Samples keep the scan's colour space (an ICC profile, CalRGB...):
        # the layers reuse it, and the ink colours are set in it too.
        space = image.get("/ColorSpace")
        calibrated = not isinstance(space, pikepdf.Name)
        device = Name.DeviceGray if layers.grayscale else Name.DeviceRGB
        background = _image(
            pdf, layers.background, layers.bg_size,
            filter_name=Name.JPXDecode if layers.codec == "/JPXDecode" else Name.DCTDecode,
            colorspace=space if calibrated else device)
        resources = pikepdf.Dictionary(Bg=background)
        content = "q /Bg Do Q"
        for slot, item in enumerate(layers.inks):
            ink_params, ink_data, ink_filter = None, item.mask, Name.FlateDecode
            if position in symbol_streams:
                ink_data, shared = symbol_streams[position]
                ink_filter = Name.JBIG2Decode
                if shared is not None:
                    ink_params = pikepdf.Dictionary(JBIG2Globals=shared)
            elif item.group4:
                ink_filter = Name.CCITTFaxDecode
                ink_params = pikepdf.Dictionary(K=-1, Columns=layers.mask_size[0],
                                                Rows=layers.mask_size[1], BlackIs1=True)
            position += 1
            ink = _image(pdf, ink_data, layers.mask_size, filter_name=ink_filter,
                         mask=True, parms=ink_params)
            resources[f"/Ink{slot}"] = ink
            if layers.foreground is not None:
                # The colour image is RGB; it keeps a calibrated RGB space.
                rgb_space = space if calibrated and not layers.grayscale else Name.DeviceRGB
                fg = _image(pdf, layers.foreground[0], layers.foreground[1],
                            filter_name=Name.DCTDecode, colorspace=rgb_space)
                fg["/Mask"] = ink
                resources.Fg = fg
                content += " q /Fg Do Q"
                continue
            content += f" q {_fill_colour(item.color, calibrated, layers.grayscale)}" \
                       f" /Ink{slot} Do Q"
        form_resources = pikepdf.Dictionary(XObject=resources)
        if calibrated:
            form_resources.ColorSpace = pikepdf.Dictionary(CS0=space)
        form = _stream(pdf, content.encode("ascii"), Type=Name.XObject,
                       Subtype=Name.Form, FormType=1, BBox=[0, 0, 1, 1],
                       Resources=form_resources)
        form[MRC_FORM_KEY] = True
        for number, _, name in users:
            # Give the page its own resource dictionaries so that sibling
            # pages (shared or inherited resources) are unaffected.
            page = pdf.pages[number - 1].obj
            own = pikepdf.Dictionary(dict(page_resources(page).items()))
            table = pikepdf.Dictionary(dict(own.XObject.items()))
            own.XObject = table
            page.Resources = own
            records.append(Record(number, table, name, image, layers.minimum))
            table[name] = form
    return records


class _Inline:
    """A context manager that mimics an executor without processes."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def submit(function, *args):
        from concurrent.futures import Future
        future = Future()
        future.set_result(function(*args))
        return future


def build(source: Path, target: Path, scratch: Path, minimum: float, jobs: int,
          gate: VisualGate, log=None, *, then=None, symbols: bool = True,
          relaxed: float | None = None) -> tuple[int, int]:
    """Layer the scans of ``source`` into ``target``; returns (passed, rejected).

    Text pages are held to ``relaxed`` (when given), other pages to
    ``minimum``. ``then(pdf)`` may modify the verified layered document
    further (the JPEG stage).
    """
    log = log or (lambda message: None)
    with open_pdf(source) as pdf:
        records = add_layers(pdf, scratch, minimum, jobs, log, symbols=symbols,
                             relaxed=relaxed)
        if not records:
            raise ValueError("no page is suitable for layered (MRC) compression")
        lossless.save(pdf, target, deflated=True)
        failed: set[int] = set()
        for threshold in sorted({r.minimum for r in records}):
            pages = [r.page for r in records if r.minimum == threshold]
            failed.update(gate.failures(target, pages, minimum=threshold))
        passed = len(records) - len(failed)
        if not passed:
            lowest = min(r.minimum for r in records)
            raise ValueError(f"no MRC page reached SSIM {lowest:.2f}")
        if failed:
            for record in records:
                if record.page in failed:
                    restore(record)
            lossless.save(pdf, target, deflated=True)
        text = sum(r.minimum < minimum and r.page not in failed for r in records)
        log(f"MRC: accepted {passed}, rejected {len(failed)} pages" +
            (f" (text threshold {relaxed:.2f}: {text})" if text else ""))
        if then is not None:
            then(pdf)
        return passed, len(failed)
