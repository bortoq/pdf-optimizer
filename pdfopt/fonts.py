"""Cutting embedded TrueType fonts down to the glyphs a document shows.

Office and PDF printers often embed whole fonts (or "subsets" that are not
cut at all): a 329 KB Calibri for a handful of characters. The text drawing
operators of every page, form, annotation appearance, tiling pattern, soft
mask and Type 3 glyph are read to learn which character codes each font
shows; the font program is then subset with fontTools keeping the glyph
ids, so content streams stay byte-identical. Pages showing a changed font
are verified pixel-for-pixel by the lossless stage.

Anything unclear leaves a font alone: a content stream that cannot be
parsed, an AcroForm (field appearances may be regenerated from its fonts),
a Type0 font with an encoding other than Identity, or a font program also
used by a font dictionary that was not found in the walked resources.
"""

from __future__ import annotations

from collections import defaultdict
import io
import logging
import zlib

from fontTools import subset as ftsubset
from fontTools import ttLib
from fontTools.agl import toUnicode
from fontTools.ttLib import TTFont
import pikepdf
from pikepdf import Name

from .pdfmodel import page_resources

# fontTools warns about every slightly malformed table of a font; the font is
# still usable, and the cut is verified pixel-for-pixel, so the noise is dropped.
logging.getLogger("fontTools").setLevel(logging.ERROR)

# WinAnsiEncoding is cp1252; its five holes decode to nothing.
_WINANSI = {code: char for code in range(256)
            if (char := bytes([code]).decode("cp1252", "ignore"))}
_SHOW = {"Tj", "'", '"', "TJ"}


class _Unparsed(Exception):
    """A content stream could not be read: font use is unknown."""


class _Usage:
    """Codes shown by each font dictionary, and where."""

    def __init__(self):
        self.codes: dict[tuple, set[int]] = defaultdict(set)
        self.pages: dict[tuple, set[int]] = defaultdict(set)
        self.known: dict[tuple, pikepdf.Object] = {}     # fonts in walked resources
        self.unsupported: set[tuple] = set()
        self._walked: set[tuple] = set()
        self._parsed: dict = {}           # a shared form is parsed once
        # Programs of direct font dictionaries (no objgen to key their use by).
        self.direct: set[tuple] = set()

    def resources(self, resources, page: int) -> None:
        """Register the fonts of a resource dictionary and walk the content
        that is not reached through ``Do`` (patterns, soft masks, Type 3)."""
        if resources is None:
            return
        for name in list((resources.get("/Font") or {}).keys()):
            font = resources.Font[name]
            if not isinstance(font, pikepdf.Dictionary):
                continue
            if not font.is_indirect:
                stream = _font_file(font)
                if stream is not None:
                    self.direct.add(stream.objgen)
                continue
            self.known[font.objgen] = font
            if font.get("/Subtype") == "/Type3":
                for glyph in (font.get("/CharProcs") or {}).values():
                    self.content(glyph, font.get("/Resources", resources), page, None)
        for pattern in (resources.get("/Pattern") or {}).values():
            if isinstance(pattern, pikepdf.Stream):          # tiling pattern
                self.content(pattern, pattern.get("/Resources"), page, None)
        for state in (resources.get("/ExtGState") or {}).values():
            mask = state.get("/SMask") if isinstance(state, pikepdf.Dictionary) else None
            group = mask.get("/G") if isinstance(mask, pikepdf.Dictionary) else None
            if isinstance(group, pikepdf.Stream):
                self.content(group, group.get("/Resources", resources), page, None)
        for xobject in (resources.get("/XObject") or {}).values():
            if isinstance(xobject, pikepdf.Stream) and xobject.get("/Subtype") == "/Form":
                self.content(xobject, xobject.get("/Resources", resources), page, None)

    def content(self, owner, resources, page: int, font) -> None:
        key = (owner.objgen if owner.is_indirect else id(owner), page,
               font.objgen if font is not None else None)
        if key in self._walked:
            return
        self._walked.add(key)
        self.resources(resources, page)
        instructions = self._parsed.get(key[0])
        if instructions is None:
            try:
                instructions = pikepdf.parse_content_stream(owner)
            except (pikepdf.PdfError, ValueError, TypeError) as exc:
                raise _Unparsed(str(exc)) from exc
            self._parsed[key[0]] = instructions
        fonts = resources.get("/Font") if resources is not None else None
        xobjects = resources.get("/XObject") if resources is not None else None
        saved = []
        for item in instructions:
            if isinstance(item, pikepdf.ContentStreamInlineImage):
                continue
            operator, operands = str(item.operator), item.operands
            if operator == "q":
                saved.append(font)
            elif operator == "Q":
                font = saved.pop() if saved else font
            elif operator == "Tf" and operands:
                font = fonts.get(operands[0]) if fonts is not None else None
            elif operator in _SHOW and font is not None:
                self._show(font, operator, operands, page)
            elif operator == "Do" and operands and xobjects is not None:
                form = xobjects.get(operands[0])
                if isinstance(form, pikepdf.Stream) and form.get("/Subtype") == "/Form":
                    self.content(form, form.get("/Resources", resources), page, font)

    def _show(self, font, operator: str, operands, page: int) -> None:
        if operator == "TJ":
            strings = [item for item in operands[0] if isinstance(item, pikepdf.String)]
        else:
            strings = [operands[-1]]
        if not font.is_indirect:
            return
        key = font.objgen
        self.pages[key].add(page)
        if font.get("/Subtype") == "/Type0":
            if str(font.get("/Encoding")) not in ("/Identity-H", "/Identity-V"):
                self.unsupported.add(key)
                return
            for string in strings:
                data = bytes(string)
                self.codes[key].update(data[i] << 8 | data[i + 1]
                                       for i in range(0, len(data) - 1, 2))
        else:
            for string in strings:
                self.codes[key].update(bytes(string))


def _descriptor(font):
    if font.get("/Subtype") == "/Type0":
        descendants = font.get("/DescendantFonts")
        font = descendants[0] if descendants else None
    return font.get("/FontDescriptor") if font is not None else None


def _font_file(font):
    descriptor = _descriptor(font)
    if isinstance(descriptor, pikepdf.Dictionary):
        stream = descriptor.get("/FontFile2")
        if isinstance(stream, pikepdf.Stream):
            return stream
    return None


def _glyphs(font, codes: set[int], tt: TTFont) -> set[int]:
    """Glyph ids the codes of one font may reach (a safe superset)."""
    order = tt.getGlyphOrder()
    index = {name: number for number, name in enumerate(order)}
    gids = {0}
    if font.get("/Subtype") == "/Type0":
        mapping = font.DescendantFonts[0].get("/CIDToGIDMap", Name.Identity)
        if isinstance(mapping, pikepdf.Stream):
            table = bytes(mapping.read_bytes())
            gids.update(table[2 * c] << 8 | table[2 * c + 1]
                        for c in codes if 2 * c + 1 < len(table))
        else:
            gids.update(codes)            # Identity: code = CID = glyph id
        return {g for g in gids if g < len(order)}
    # A simple TrueType font: every route a viewer may take from a code.
    differences = {}
    encoding = font.get("/Encoding")
    if isinstance(encoding, pikepdf.Dictionary) and "/Differences" in encoding:
        code = 0
        for item in encoding.Differences:
            if isinstance(item, int):
                code = int(item)
            else:
                differences[code] = str(item)[1:]
                code += 1
    cmaps = [(table.platformID, table.platEncID, table.cmap)
             for table in tt["cmap"].tables] if "cmap" in tt else []
    for code in codes:
        names = []
        unicode = _WINANSI.get(code)
        if code in differences:
            names.append(differences[code])
            unicode = toUnicode(differences[code]) or unicode
        for platform, encoding_id, cmap in cmaps:
            for probe in (code, 0xF000 + code, 0xF100 + code, 0xF200 + code):
                if probe in cmap:
                    names.append(cmap[probe])
            if (unicode and (platform, encoding_id) in ((3, 1), (3, 10), (0, 3), (0, 4))
                    and ord(unicode[0]) in cmap):
                names.append(cmap[ord(unicode[0])])
        found = {index[name] for name in names if name in index}
        if not found and code < len(order):
            gids.add(code)                # no cmap route: viewers use the code as glyph id
        gids |= found
    return gids


def _restore_cmap(tt: TTFont, tables) -> None:
    from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
    if "cmap" not in tt:
        tt["cmap"] = ttLib.newTable("cmap")
        tt["cmap"].tableVersion = 0
    cmap = tt["cmap"]
    cmap.tables = []
    for platform, encoding_id, format_, language, entries in tables:
        if not entries or (format_ == 0 and max(entries) > 255):
            continue
        table = CmapSubtable.newSubtable(format_)
        table.platformID, table.platEncID, table.language = platform, encoding_id, language
        table.cmap = entries
        cmap.tables.append(table)


def _subset(data: bytes, gids: set[int]) -> bytes:
    tt = TTFont(io.BytesIO(data))
    order = tt.getGlyphOrder()
    kept = {order[g] for g in gids if g < len(order)}
    # The subsetter keeps only the cmap entries of requested Unicode values and
    # empties tables such as the (1, 0) one of LibreOffice fonts, which maps
    # codes straight to glyphs; viewers then lose the way from code to glyph.
    tables = [(t.platformID, t.platEncID, t.format, t.language,
               {code: name for code, name in t.cmap.items() if name in kept})
              for t in tt["cmap"].tables] if "cmap" in tt else []
    options = ftsubset.Options()
    options.retain_gids = True           # content streams keep their codes
    options.glyph_names = True
    options.notdef_outline = True
    options.name_IDs = ["*"]
    options.name_languages = ["*"]
    options.layout_features = ["*"]
    options.drop_tables = []             # only glyphs go; hinting and tables stay
    options.hinting = True
    options.recalc_bounds = False
    options.recalc_timestamp = False
    subsetter = ftsubset.Subsetter(options)
    subsetter.populate(gids=sorted(gids))
    subsetter.subset(tt)
    if tables:
        _restore_cmap(tt, tables)
    output = io.BytesIO()
    tt.save(output)
    return output.getvalue()


def subset_fonts(pdf: pikepdf.Pdf) -> tuple[set[int], int]:
    """Cut embedded TrueType programs to the glyphs in use, in place.

    Returns the zero-based pages that show a changed font (to be verified
    pixel-for-pixel) and the number of font programs cut.
    """
    if "/AcroForm" in pdf.Root:
        return set(), 0
    usage = _Usage()
    try:
        for number, page in enumerate(pdf.pages):
            usage.content(page.obj, page_resources(page.obj), number, None)
            for annotation in page.obj.get("/Annots") or []:
                appearance = annotation.get("/AP") if isinstance(
                    annotation, pikepdf.Dictionary) else None
                for state in (appearance or {}).values():
                    streams = [state] if isinstance(state, pikepdf.Stream) else \
                        [s for s in (state or {}).values() if isinstance(s, pikepdf.Stream)]
                    for stream in streams:
                        usage.content(stream, stream.get("/Resources"), number, None)
    except (_Unparsed, RecursionError):
        return set(), 0

    # Font programs with every user known (shared programs need all users).
    # Font dictionaries are recognized by subtype (/Type is often missing);
    # CIDFonts are reached through their Type0 parents.
    users: dict[tuple, list] = defaultdict(list)
    unsafe: set[tuple] = set(usage.direct)
    fonts = [obj for obj in pdf.objects if isinstance(obj, pikepdf.Dictionary)
             and obj.get("/Subtype") in ("/TrueType", "/Type0")]
    for font in fonts:
        stream = _font_file(font)
        if stream is None:
            continue
        if font.objgen in usage.known and font.objgen not in usage.unsupported:
            users[stream.objgen].append(font)
        else:
            unsafe.add(stream.objgen)

    pages: set[int] = set()
    count = 0
    for key, fonts in users.items():
        if key in unsafe:
            continue
        stream = _font_file(fonts[0])
        try:
            data = bytes(stream.read_bytes())
            tt = TTFont(io.BytesIO(data))
            gids: set[int] = set()
            for font in fonts:
                gids |= _glyphs(font, usage.codes.get(font.objgen, set()), tt)
            cut = _subset(data, gids)
        except Exception:     # a font fontTools cannot handle stays as it is
            continue
        packed = zlib.compress(cut, 9)
        if len(packed) >= len(stream.read_raw_bytes()):
            continue
        stream.write(packed, filter=Name.FlateDecode)
        stream.Length1 = len(cut)
        count += 1
        for font in fonts:
            pages |= usage.pages.get(font.objgen, set())
    return pages, count
