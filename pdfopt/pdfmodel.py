"""Reading PDFs: safety checks, page fingerprints, images and the marker."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re

import pikepdf

from .tools import ToolError, last_line, require, run

MARKER_KEY = "/PdfOpt"
MARKER_VERSION = 1
MRC_FORM_KEY = "/PdfOptLayers"
EMPTY_DIGEST = hashlib.sha256(b"").digest()


class Skipped(ValueError):
    """The file is intentionally left alone (not an error)."""


def open_pdf(path: Path) -> pikepdf.Pdf:
    try:
        pdf = pikepdf.open(path)
    except pikepdf.PasswordError as exc:
        raise Skipped("password required") from exc
    except (pikepdf.PdfError, OSError) as exc:
        raise ValueError(f"PDF: cannot read the source: {exc}") from exc
    if pdf.is_encrypted:
        pdf.close()
        raise Skipped("encrypted file (--decrypt removes the encryption)")
    return pdf


def has_signature(pdf: pikepdf.Pdf) -> bool:
    """True for documents with a signature or a certification dictionary.

    Signature dictionaries are indirect objects carrying ``/ByteRange``;
    scanning objects (not file bytes) avoids matching page text that merely
    mentions the key.
    """
    root = pdf.Root
    if "/Perms" in root:
        return True
    form = root.get("/AcroForm")
    if form is not None and int(form.get("/SigFlags", 0) or 0) & 1:
        return True
    return any(isinstance(obj, (pikepdf.Dictionary, pikepdf.Stream))
               and "/ByteRange" in obj for obj in pdf.objects)


MAX_TREE_DEPTH = 64     # page trees are a few levels deep; a longer chain is a loop


def _inherited(node, key: str, default=None):
    """An inheritable page attribute, looked up through ``/Parent``.

    A damaged file may link ``/Parent`` in a circle, so the climb is bounded.
    """
    for _ in range(MAX_TREE_DEPTH):
        if not isinstance(node, pikepdf.Dictionary):
            break
        if key in node:
            return node[key]
        node = node.get("/Parent")
    return default


def page_resources(page_obj):
    return _inherited(page_obj, "/Resources")


_TEXT_OBJECT = re.compile(rb"(?:^|\s)BT(?:\s|$)")


def _forms_have_text(resources, seen: set) -> bool:
    """A ``BT`` text object in any Form XObject reachable from ``resources``
    (undecodable forms count as text)."""
    xobjects = resources.get("/XObject") if resources is not None else None
    if xobjects is None:
        return False
    for name in list(xobjects.keys()):
        form = xobjects[name]
        if form.get("/Subtype") != "/Form" or form.objgen in seen:
            continue
        seen.add(form.objgen)
        try:
            if _TEXT_OBJECT.search(bytes(form.read_bytes())):
                return True
        except (pikepdf.PdfError, ValueError):
            return True
        if _forms_have_text(form.get("/Resources"), seen):
            return True
    return False


def _content_facts(page_obj) -> tuple[bytes, bool]:
    """Digest of the page content and whether it has text: a ``BT`` text
    object in the content or its forms. Unused fonts among the resources do
    not count; undecodable content does."""
    text = _forms_have_text(page_resources(page_obj), set())
    contents = page_obj.get("/Contents")
    if contents is None:
        return EMPTY_DIGEST, text
    streams = list(contents) if isinstance(contents, pikepdf.Array) else [contents]
    digest = hashlib.sha256()
    for stream in streams:
        try:
            data = bytes(stream.read_bytes())
            text = text or _TEXT_OBJECT.search(data) is not None
        except (pikepdf.PdfError, ValueError):
            data = bytes(stream.read_raw_bytes())
            text = True
        digest.update(data)
        digest.update(b"\n")
    return digest.digest(), text


def _box(value) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(round(float(v), 2) for v in value)


# Document-level features a rewrite by an external engine may drop; a candidate
# has to keep every one the source has (their presence is compared).
CATALOG_KEYS = ("/Outlines", "/PageLabels", "/Lang", "/ViewerPreferences",
                "/OutputIntents", "/StructTreeRoot", "/MarkInfo", "/OCProperties",
                "/PageLayout", "/PageMode", "/OpenAction")

_PDFA_PART = re.compile(rb"pdfaid:part\s*(?:=\s*[\"']|>)\s*(\d)")


def pdfa_part(pdf: pikepdf.Pdf) -> int | None:
    """The PDF/A part the XMP metadata claims (1, 2, 3...), or None."""
    try:
        xmp = pdf.Root.get("/Metadata")
        found = _PDFA_PART.search(bytes(xmp.read_bytes())) if xmp is not None else None
    except (pikepdf.PdfError, ValueError, TypeError, AttributeError):
        return None
    return int(found.group(1)) if found else None


@dataclass(frozen=True)
class PageFacts:
    box: tuple | None
    crop: tuple | None
    rotation: int
    annotations: int
    content: bytes
    text: bool = False


@dataclass
class Fingerprint:
    """What must survive optimization: layout, content and metadata."""

    path: Path
    pages: tuple[PageFacts, ...]
    metadata: tuple
    interactive: bool
    catalog: frozenset = frozenset()     # CATALOG_KEYS the document has
    pdfa: int | None = None              # the PDF/A part it claims
    _text: tuple | None = field(default=None, repr=False)

    @classmethod
    def read(cls, pdf: pikepdf.Pdf, path: Path) -> Fingerprint:
        pages = []
        for page in pdf.pages:
            obj = page.obj
            annots = obj.get("/Annots")
            digest, text = _content_facts(obj)
            pages.append(PageFacts(
                _box(_inherited(obj, "/MediaBox")),
                _box(_inherited(obj, "/CropBox") or _inherited(obj, "/MediaBox")),
                int(_inherited(obj, "/Rotate", 0) or 0) % 360,
                len(annots) if annots is not None else 0,
                digest, text))
        metadata = tuple(sorted((str(k), str(v)) for k, v in pdf.docinfo.items()))
        xmp = pdf.Root.get("/Metadata")
        if xmp is not None:
            metadata += (("xmp", hashlib.sha256(xmp.read_bytes()).hexdigest()),)
        root = pdf.Root
        interactive = ("/AcroForm" in root or "/Names" in root or
                       any(page.annotations for page in pages))
        catalog = frozenset(key for key in CATALOG_KEYS if key in root)
        return cls(path, tuple(pages), metadata, interactive, catalog, pdfa_part(pdf))

    @property
    def has_text_layer(self) -> bool:
        """Any page with text: Ghostscript is only offered to documents
        without one (its candidates are still checked page by page)."""
        return any(page.text for page in self.pages)

    def text(self) -> tuple:
        """Per-page text digests (slow; only needed when content streams
        legitimately change, e.g. after Ghostscript)."""
        if self._text is None:
            self._text = _text_digests(self.path)
        return self._text


def _text_digests(path: Path) -> tuple:
    """A digest of the text of every page, as poppler's ``pdftotext`` reads it
    (pages end with a form feed)."""
    done = run([require("pdftotext"), "-enc", "UTF-8", str(path), "-"], binary=True)
    if done.returncode != 0:
        raise ToolError("pdftotext: " + last_line(done))
    pages = done.stdout.split(b"\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return tuple(hashlib.sha256(b" ".join(page.split())).digest() for page in pages)


def compare(candidate: Fingerprint, baseline: Fingerprint, *,
            content: bool, metadata: bool) -> None:
    """Raise ValueError if the candidate differs from the baseline.

    ``content`` demands byte-identical page content streams; without it a
    difference falls back to comparing the extracted text.
    """
    if len(candidate.pages) != len(baseline.pages):
        raise ValueError("PDF: the page count changed")
    texts_needed = False
    for new, old in zip(candidate.pages, baseline.pages):
        if (new.box, new.crop, new.rotation, new.annotations) != (
                old.box, old.crop, old.rotation, old.annotations):
            raise ValueError("PDF: page geometry or annotations changed")
        if new.content != old.content:
            texts_needed = True
    if texts_needed:
        if content:
            raise ValueError("PDF: page content changed in lossless mode")
        if candidate.text() != baseline.text():
            raise ValueError("PDF: page text changed")
    lost = sorted(baseline.catalog - candidate.catalog)
    if lost:
        raise ValueError("PDF: the document lost " + ", ".join(key[1:] for key in lost))
    if metadata and candidate.metadata != baseline.metadata:
        raise ValueError("PDF: metadata changed in lossless mode")


def read_marker(pdf: pikepdf.Pdf) -> tuple[int, bool] | None:
    marker = pdf.Root.get(MARKER_KEY)
    if marker is None:
        return None
    try:
        return int(marker.get("/Version", 0)), bool(marker.get("/Lossy", False))
    except (TypeError, ValueError, AttributeError, pikepdf.PdfError):
        return None


def marker_quality(pdf: pikepdf.Pdf) -> int | None:
    """The ``-q`` value (percent) a lossy file was made with, if recorded."""
    marker = pdf.Root.get(MARKER_KEY)
    try:
        value = marker.get("/Quality") if marker is not None else None
        return int(value) if value is not None else None
    except (TypeError, ValueError, AttributeError, pikepdf.PdfError):
        return None


def set_marker(pdf: pikepdf.Pdf, lossy: bool, quality: int | None = None) -> None:
    """Mark the document as optimized. ``quality`` is the ``-q`` percent of a
    lossy run; the lowest one the file ever went through is kept."""
    previous = read_marker(pdf)
    earlier = marker_quality(pdf)
    lossy = lossy or bool(previous and previous[1])
    marker = pikepdf.Dictionary(Version=MARKER_VERSION, Lossy=lossy)
    if lossy and quality is not None:
        marker.Quality = min(quality, earlier) if earlier is not None else quality
    elif earlier is not None:
        marker.Quality = earlier
    pdf.Root[MARKER_KEY] = marker


def resources_images(resources, cache: dict, active: set) -> list:
    """Image XObjects reachable from a resource dictionary (forms included).

    Forms created by the MRC stage are skipped so lossy passes never touch
    their own layers twice.
    """
    found: list = []
    xobjects = resources.get("/XObject") if resources is not None else None
    if xobjects is None:
        return found
    for name in list(xobjects.keys()):
        obj = xobjects[name]
        subtype = obj.get("/Subtype")
        if subtype == "/Image":
            found.append(obj)
        elif subtype == "/Form" and MRC_FORM_KEY not in obj:
            key = obj.objgen
            if key in cache:
                found.extend(cache[key])
            elif key not in active:
                active.add(key)
                inner = resources_images(obj.get("/Resources", resources),
                                         cache, active)
                active.discard(key)
                cache[key] = inner
                found.extend(inner)
    return found


@dataclass
class ImageRef:
    obj: pikepdf.Object
    pages: set[int]


def collect_images(pdf: pikepdf.Pdf) -> dict[tuple, ImageRef]:
    """Unique image XObjects with the zero-based pages that show them."""
    images: dict[tuple, ImageRef] = {}
    cache: dict = {}
    for number, page in enumerate(pdf.pages):
        for obj in resources_images(page_resources(page.obj), cache, set()):
            ref = images.setdefault(obj.objgen, ImageRef(obj, set()))
            ref.pages.add(number)
    return images


def iter_streams(pdf: pikepdf.Pdf) -> Iterator[pikepdf.Stream]:
    for obj in pdf.objects:
        if isinstance(obj, pikepdf.Stream):
            yield obj


def filters_of(obj) -> list[str]:
    value = obj.get("/Filter")
    if value is None:
        return []
    if isinstance(value, pikepdf.Array):
        return [str(item) for item in value]
    return [str(value)]
