"""Type 1 font programs repacked as CFF (the compact font format).

A Type 1 program in a PDF keeps its glyphs in an encrypted section that Flate
cannot compress: a 20 KB font stays 20 KB. The same outlines, widths, stem
hints and flex as a CFF font (``/FontFile3`` with ``/Subtype /Type1C``) take a
fifth of that. pdfTeX and dvips embed Type 1 fonts, so in a typical paper
they are most of the file.

Every glyph program is translated operator by operator:

* outlines: the path operators are the same in both formats; the left
  sidebearing of ``hsbw`` moves into the first ``moveto``;
* stem hints become ``hstem``/``vstem``; glyphs that replace their hints on
  the way (OtherSubr 3) get ``hstemhm``/``vstemhm`` and a ``hintmask`` at
  every replacement;
* flex (OtherSubrs 0-2) becomes the ``flex`` operator;
* accented glyphs made with ``seac`` are written out as the two outlines
  (without hints).

Anything else (an unknown OtherSubr, a vertical advance, a stroked font)
leaves the font as it is. Pages that show a converted font are compared
pixel for pixel by the lossless stage, like every other change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io

from fontTools.encodings.StandardEncoding import StandardEncoding
from fontTools.fontBuilder import FontBuilder
from fontTools.misc import eexec, psLib
from fontTools.misc.psCharStrings import T1CharString, T2CharString

MAX_STEMS = 96          # the limit of the Type 2 format
STEMS_PER_OPERATOR = 22  # two arguments each, under the stack limit of 48
MAX_DEPTH = 10          # subroutine nesting
_PRIVATE_ARRAYS = ("BlueValues", "OtherBlues", "FamilyBlues", "FamilyOtherBlues",
                   "StemSnapH", "StemSnapV")
_PRIVATE_NUMBERS = ("BlueScale", "BlueShift", "BlueFuzz", "LanguageGroup",
                    "ExpansionFactor")


class Unsupported(ValueError):
    """The font uses something this conversion does not cover."""


def _array(value) -> list | None:
    """A PostScript array ([...] or {...}) of numbers as a list, else None."""
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(item, (int, float)) for item in value):
        return [_number(item) for item in value]
    return None


def _number(value):
    """Integers stay integers (shorter, exact); other values become floats."""
    if isinstance(value, float) and value == int(value):
        return int(value)
    return value


@dataclass
class _Outline:
    """One translated glyph: where it starts, its path, and its hints."""

    width: float = 0
    sbx: float = 0
    start: tuple[float, float] | None = None    # absolute first point
    end: tuple[float, float] = (0, 0)
    body: list = field(default_factory=list)    # path after the first moveto
    # Hint groups: (position in body, horizontal stems, vertical stems).
    groups: list[tuple[int, set, set]] = field(default_factory=lambda: [(0, set(), set())])
    composite: bool = False


class _Translator:
    """Runs the Type 1 programs of one font and writes Type 2 programs."""

    def __init__(self, charstrings: dict, subrs: list):
        self.charstrings = charstrings
        self.subrs = subrs
        self.done: dict[str, _Outline] = {}
        self.active: set[str] = set()

    # -- running one glyph ---------------------------------------------------

    def outline(self, name: str) -> _Outline:
        if name in self.done:
            return self.done[name]
        if name in self.active or name not in self.charstrings:
            raise Unsupported(f"glyph {name}")
        self.active.add(name)
        glyph = _Outline()
        state = _State(glyph)
        self._run(self.charstrings[name], state, 0)
        glyph.end = (state.x, state.y)
        self.active.discard(name)
        self.done[name] = glyph
        return glyph

    def _run(self, charstring: T1CharString, state: _State, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise Unsupported("subroutines nested too deep")
        charstring.decompile()
        for token in charstring.program:
            if state.finished:
                return
            if not isinstance(token, str):
                state.stack.append(token)
                continue
            if token == "callsubr":
                index = state.pop()
                if not isinstance(index, int) or not 0 <= index < len(self.subrs):
                    raise Unsupported("subroutine number")
                self._run(self.subrs[index], state, depth + 1)
            elif token == "return":
                return
            elif token == "seac":
                self._seac(state)
            else:
                state.operator(token)

    def _seac(self, state: _State) -> None:
        """An accented character: the outlines of two other glyphs."""
        if len(state.stack) < 5 or state.glyph.start is not None:
            raise Unsupported("seac")
        asb, adx, ady, base_code, accent_code = state.stack[-5:]
        state.stack.clear()
        try:
            base = self.outline(StandardEncoding[int(base_code)])
            accent = self.outline(StandardEncoding[int(accent_code)])
        except (IndexError, ValueError) as exc:
            raise Unsupported("seac") from exc
        glyph = state.glyph
        glyph.composite = True
        glyph.groups = [(0, set(), set())]
        if base.start is not None:
            glyph.start = base.start
            glyph.body = list(base.body)
            state.x, state.y = base.end
        if accent.start is not None:
            # The accent's sidebearing point goes to (adx, ady) from the
            # sidebearing point of this character.
            x = accent.start[0] + adx - asb + glyph.sbx
            y = accent.start[1] + ady
            if glyph.start is None:
                glyph.start = (x, y)
            else:
                glyph.body += [_number(x - state.x), _number(y - state.y), "rmoveto"]
            glyph.body += accent.body
            state.x = accent.end[0] + adx - asb + glyph.sbx
            state.y = accent.end[1] + ady
        state.finished = True

    # -- writing -------------------------------------------------------------

    def program(self, name: str) -> list:
        glyph = self.outline(name)
        out: list = []
        if glyph.width:
            out.append(_number(glyph.width))
        horizontal = sorted({stem for _, h, _ in glyph.groups for stem in h})
        vertical = sorted({stem for _, _, v in glyph.groups for stem in v})
        hinted = (not glyph.composite and (horizontal or vertical)
                  and len(horizontal) + len(vertical) <= MAX_STEMS)
        masked = hinted and len(glyph.groups) > 1
        if hinted:
            out += _stems(horizontal, "hstemhm" if masked else "hstem")
            out += _stems(vertical, "vstemhm" if masked else "vstem")
        order = horizontal + vertical       # the bits of a hint mask

        def mask(group) -> list:
            active = [stem in group[1] for stem in horizontal] + \
                     [stem in group[2] for stem in vertical]
            bits = bytearray((len(order) + 7) // 8)
            for position, on in enumerate(active):
                if on:
                    bits[position // 8] |= 0x80 >> (position % 8)
            return ["hintmask", bytes(bits)]

        if masked:
            out += mask(glyph.groups[0])
        if glyph.start is not None:
            out += [_number(glyph.start[0]), _number(glyph.start[1]), "rmoveto"]
            if masked:
                position = 0
                for group in glyph.groups[1:]:
                    out += glyph.body[position:group[0]]
                    out += mask(group)
                    position = group[0]
                out += glyph.body[position:]
            else:
                out += glyph.body
        out.append("endchar")
        return out


def _stems(stems: list, operator: str) -> list:
    """Stem hints as Type 2 wants them: sorted, each edge relative to the last."""
    out: list = []
    edge = 0
    for count, (position, width) in enumerate(stems):
        if count and count % STEMS_PER_OPERATOR == 0:
            out.append(operator)
        out += [_number(position - edge), _number(width)]
        edge = position + width
    if stems:
        out.append(operator)
    return out


class _State:
    """The Type 1 interpreter state while one glyph runs."""

    def __init__(self, glyph: _Outline):
        self.glyph = glyph
        self.stack: list = []
        self.results: list = []         # what OtherSubrs hand back to ``pop``
        self.x = self.y = 0.0
        self.flex: list | None = None
        self.finished = False
        self.sby = 0.0

    def pop(self):
        if not self.stack:
            raise Unsupported("stack underflow")
        return self.stack.pop()

    def take(self, count: int) -> list:
        if len(self.stack) < count:
            raise Unsupported("stack underflow")
        values = self.stack[-count:]
        del self.stack[-count:]
        return values

    def operator(self, name: str) -> None:
        handler = getattr(self, "op_" + name, None)
        if handler is None:
            raise Unsupported(f"operator {name}")
        handler()

    # -- metrics -------------------------------------------------------------

    def op_hsbw(self) -> None:
        self.glyph.sbx, self.glyph.width = self.take(2)
        self.x, self.y = self.glyph.sbx, 0
        self.stack.clear()

    def op_sbw(self) -> None:
        sbx, sby, width, height = self.take(4)
        if height:
            raise Unsupported("vertical advance")
        self.glyph.sbx, self.glyph.width, self.sby = sbx, width, sby
        self.x, self.y = sbx, sby
        self.stack.clear()

    # -- path ----------------------------------------------------------------

    def _move(self, dx, dy) -> None:
        self.stack.clear()
        if self.flex is not None:
            self.flex.append((dx, dy))
            self.x += dx
            self.y += dy
            return
        self.x += dx
        self.y += dy
        if self.glyph.start is None:
            self.glyph.start = (self.x, self.y)
        elif dy == 0:
            self.glyph.body += [_number(dx), "hmoveto"]
        elif dx == 0:
            self.glyph.body += [_number(dy), "vmoveto"]
        else:
            self.glyph.body += [_number(dx), _number(dy), "rmoveto"]

    def op_rmoveto(self) -> None:
        self._move(*self.take(2))

    def op_hmoveto(self) -> None:
        self._move(self.pop(), 0)

    def op_vmoveto(self) -> None:
        self._move(0, self.pop())

    def _draw(self, operator: str, values: list, dx, dy) -> None:
        if self.glyph.start is None or self.flex is not None:
            raise Unsupported("path without a starting point")
        self.stack.clear()
        self.x += dx
        self.y += dy
        self.glyph.body += [*map(_number, values), operator]

    def op_rlineto(self) -> None:
        dx, dy = values = self.take(2)
        self._draw("rlineto", values, dx, dy)

    def op_hlineto(self) -> None:
        dx = self.pop()
        self._draw("hlineto", [dx], dx, 0)

    def op_vlineto(self) -> None:
        dy = self.pop()
        self._draw("vlineto", [dy], 0, dy)

    def op_rrcurveto(self) -> None:
        values = self.take(6)
        self._draw("rrcurveto", values, values[0] + values[2] + values[4],
                   values[1] + values[3] + values[5])

    def op_vhcurveto(self) -> None:
        dy1, dx2, dy2, dx3 = values = self.take(4)
        self._draw("vhcurveto", values, dx2 + dx3, dy1 + dy2)

    def op_hvcurveto(self) -> None:
        dx1, dx2, dy2, dy3 = values = self.take(4)
        self._draw("hvcurveto", values, dx1 + dx2, dy2 + dy3)

    def op_closepath(self) -> None:
        self.stack.clear()      # Type 2 closes a subpath at the next moveto

    def op_endchar(self) -> None:
        self.finished = True

    # -- hints ---------------------------------------------------------------

    def _stem(self, vertical: bool, position, width) -> None:
        position += self.glyph.sbx if vertical else self.sby
        if width < 0:
            position, width = position + width, -width
        _, horizontal, verticals = self.glyph.groups[-1]
        (verticals if vertical else horizontal).add((position, width))

    def op_hstem(self) -> None:
        self._stem(False, *self.take(2))
        self.stack.clear()

    def op_vstem(self) -> None:
        self._stem(True, *self.take(2))
        self.stack.clear()

    def op_hstem3(self) -> None:
        values = self.take(6)
        for index in (0, 2, 4):
            self._stem(False, values[index], values[index + 1])
        self.stack.clear()

    def op_vstem3(self) -> None:
        values = self.take(6)
        for index in (0, 2, 4):
            self._stem(True, values[index], values[index + 1])
        self.stack.clear()

    def op_dotsection(self) -> None:
        self.stack.clear()      # has no meaning in Type 2

    # -- arithmetic and OtherSubrs -------------------------------------------

    def op_div(self) -> None:
        dividend, divisor = self.take(2)
        if not divisor:
            raise Unsupported("division by zero")
        self.stack.append(dividend / divisor)

    def op_pop(self) -> None:
        if not self.results:
            raise Unsupported("pop without a result")
        self.stack.append(self.results.pop())

    def op_setcurrentpoint(self) -> None:
        self.take(2)

    def op_callothersubr(self) -> None:
        number = self.pop()
        count = self.pop()
        if not isinstance(count, int) or count < 0:
            raise Unsupported("OtherSubr arguments")
        arguments = self.take(count)
        if number == 1 and self.glyph.start is not None and self.flex is None:
            self.flex = []                              # flex starts
        elif number == 2 and self.flex is not None:
            pass                                        # one more flex point
        elif number == 0 and self.flex is not None and len(self.flex) == 7 and count == 3:
            (rx, ry), first, *rest = self.flex
            deltas = [first[0] + rx, first[1] + ry] + [value for point in rest for value in point]
            self.flex = None
            self.glyph.body += [*map(_number, deltas), _number(arguments[0]), "flex"]
            self.results += [arguments[2], arguments[1]]
        elif number == 3 and count == 1:
            # Hint replacement: the stems declared next replace the old ones.
            if self.glyph.start is not None:
                self.glyph.groups.append((len(self.glyph.body), set(), set()))
            self.results.append(arguments[0])
        else:
            raise Unsupported(f"OtherSubr {number}")


# ------------------------------------------------------------------ fonts --

def _parse(data: bytes) -> dict:
    """The font dictionary of a Type 1 program, glyph programs decrypted."""
    font = psLib.suckfont(data, "latin-1")
    private = font["Private"]
    skip = private.get("lenIV", 4)
    if not isinstance(skip, int) or skip < 0:
        raise Unsupported("lenIV")
    subrs = private.get("Subrs", [])
    for index, code in enumerate(subrs):
        plain, _ = eexec.decrypt(code, 4330)
        subrs[index] = T1CharString(plain[skip:], subrs=subrs)
    charstrings = font["CharStrings"]
    for name, code in charstrings.items():
        plain, _ = eexec.decrypt(code, 4330)
        charstrings[name] = T1CharString(plain[skip:], subrs=subrs)
    private["Subrs"] = subrs
    return font


def to_cff(data: bytes) -> bytes:
    """A Type 1 font program as a bare CFF font, or ``Unsupported``."""
    try:
        font = _parse(data)
        charstrings = font["CharStrings"]
        private = font["Private"]
        if font.get("PaintType", 0) != 0 or font.get("FontType", 1) != 1:
            raise Unsupported("not a filled Type 1 font")
        translator = _Translator(charstrings, private["Subrs"])
        programs = {name: translator.program(name) for name in charstrings}
    except Unsupported:
        raise
    except Exception as exc:    # a font the PostScript reader cannot follow
        raise Unsupported(f"{type(exc).__name__}: {exc}") from exc
    programs.setdefault(".notdef", ["endchar"])
    order = [".notdef"] + [name for name in programs if name != ".notdef"]

    values: dict = {"defaultWidthX": 0, "nominalWidthX": 0}
    for key in _PRIVATE_ARRAYS:
        numbers = _array(private.get(key))
        if numbers:
            values[key] = numbers
    for key in _PRIVATE_NUMBERS:
        if isinstance(private.get(key), (int, float)):
            values[key] = private[key]
    for key in ("StdHW", "StdVW"):
        numbers = _array(private.get(key))
        if numbers:
            values[key] = numbers[0]
    if private.get("ForceBold") is True:
        values["ForceBold"] = 1

    matrix = _array(font.get("FontMatrix"))
    info: dict = {"FontMatrix": [float(v) for v in matrix] if matrix and len(matrix) == 6
                  else [0.001, 0, 0, 0.001, 0, 0]}
    box = _array(font.get("FontBBox"))
    if box and len(box) == 4:
        info["FontBBox"] = box
    encoding = font.get("Encoding")
    if isinstance(encoding, (list, tuple)) and len(encoding) == 256:
        names = [name if name in programs else ".notdef" for name in encoding]
        wanted = [name if name in programs else ".notdef" for name in StandardEncoding]
        if names != wanted:
            # CFF gives every glyph one code; a font that shows one glyph under
            # two codes cannot keep its own encoding.
            coded = [name for name in names if name != ".notdef"]
            if len(set(coded)) != len(coded):
                raise Unsupported("a glyph with two codes")
            info["Encoding"] = names

    builder = FontBuilder(1000, isTTF=False)
    # The font's own bounding box, not one computed from the outlines: poppler
    # sizes its glyph bitmaps by it, and a tighter box moves glyphs by a pixel.
    builder.font.recalcBBoxes = False
    builder.setupGlyphOrder(order)
    builder.setupCFF(str(font.get("FontName", "Font")), info,
                     {name: T2CharString(program=programs[name]) for name in order}, values)
    out = io.BytesIO()
    try:
        builder.font["CFF "].cff.compile(out, builder.font, isCFF2=False)
    except Exception as exc:
        raise Unsupported(f"{type(exc).__name__}: {exc}") from exc
    return out.getvalue()
