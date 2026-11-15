"""Type 1 font programs repacked as CFF."""

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from fontTools import t1Lib
from fontTools.cffLib import CFFFontSet
from fontTools.pens.recordingPen import DecomposingRecordingPen, RecordingPen
import pikepdf

from pdfopt import fonts, lossless, type1
from pdfopt.optimize import optimize_pdf
from pdfopt.pdfmodel import open_pdf
from pdfopt.render import identical_pages

from .helpers import make_type1_pdf, need, type1_font


def path_of(pen) -> list:
    return [step for step in pen.value if step[0] not in ("closePath", "endPath")]


def converted(data: bytes):
    """The top dictionary of the CFF font made from a Type 1 program."""
    fontset = CFFFontSet()
    fontset.decompile(BytesIO(type1.to_cff(data)), None)
    return fontset.topDictIndex[0]


def program(top, name: str) -> list:
    charstring = top.CharStrings[name]
    charstring.decompile()
    return charstring.program


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.data = type1_font()[0]
        self.top = converted(self.data)

    def test_every_outline_and_width_is_kept(self):
        original = t1Lib.T1Font.__new__(t1Lib.T1Font)
        original.data, original.encoding = self.data, "latin-1"
        original.parse()
        glyphs = original.getGlyphSet()
        names = sorted(glyphs.keys())
        self.assertEqual(names, sorted(self.top.CharStrings.keys()))
        for name in names:
            with self.subTest(name):
                before = DecomposingRecordingPen(glyphs)
                glyphs[name].draw(before)
                after = RecordingPen()
                self.top.CharStrings[name].draw(after)
                self.assertEqual(path_of(after), path_of(before))
                self.assertEqual(self.top.CharStrings[name].width, glyphs[name].width)

    def test_stems_are_absolute_and_relative_to_each_other(self):
        # A: hsbw 50, "0 50 hstem", "100 60 vstem" -> the vstem moves by the
        # sidebearing.
        self.assertEqual(program(self.top, "A")[:7],
                         [600, 0, 50, "hstem", 150, 60, "vstem"])

    def test_hint_replacement_becomes_a_hint_mask(self):
        steps = program(self.top, "C")
        # Stems of both groups, sorted, each edge relative to the last.
        self.assertEqual(steps[:11], [640, 0, 40, 260, 40, "hstemhm",
                                      110, 60, 260, 60, "vstemhm"])
        masks = [steps[i + 1] for i, step in enumerate(steps) if step == "hintmask"]
        self.assertEqual(masks, [bytes([0b10100000]), bytes([0b01010000])])

    def test_flex_becomes_the_flex_operator(self):
        steps = program(self.top, "B")
        at = steps.index("flex")
        self.assertEqual(steps[at - 13:at], [-300, 20, -100, 20, -100, 0,
                                             -100, -20, -100, 0, -50, -20, 50])

    def test_an_accented_character_is_written_out(self):
        steps = program(self.top, "Agrave")
        self.assertNotIn("seac", steps)
        self.assertEqual(steps.count("rmoveto"), 2)     # base and accent

    def test_the_encoding_and_the_font_box_are_kept(self):
        encoding = {code: name for code, name in enumerate(self.top.Encoding)
                    if name != ".notdef"}
        self.assertEqual(encoding, {65: "A", 66: "B", 67: "C", 68: "Agrave",
                                    193: "grave"})
        # The box of the font, not one computed from the outlines: poppler
        # sizes glyph bitmaps by it.
        self.assertEqual(list(self.top.FontBBox), [-50, -250, 1000, 950])
        self.assertEqual(self.top.Private.BlueValues, [-15, 0, 700, 715])

    def test_unknown_features_are_refused(self):
        for label, options in {"unknown OtherSubr": {"othersubr": 9},
                               "stroked font": {"paint_type": 2},
                               "one glyph, two codes": {"duplicate_code": True}}.items():
            with self.subTest(label), self.assertRaises(type1.Unsupported):
                type1.to_cff(type1_font(**options)[0])
        with self.assertRaises(type1.Unsupported):
            type1.to_cff(b"%!PS-AdobeFont-1.0: broken\n")


def fontfile_kind(path: Path) -> list[str]:
    with open_pdf(path) as pdf:
        return sorted(key for obj in pdf.objects if isinstance(obj, pikepdf.Dictionary)
                      and obj.get("/Type") == "/FontDescriptor"
                      for key in ("/FontFile", "/FontFile3") if key in obj)


class FontStepTests(unittest.TestCase):
    def convert(self, path: Path, skip=frozenset()) -> fonts.FontChanges:
        with pikepdf.open(path, allow_overwriting_input=True) as pdf:
            changes = fonts.subset_fonts(pdf, skip)
            pdf.save(path)
        return changes

    def test_the_program_is_replaced_and_its_pages_reported(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "t1.pdf"
            make_type1_pdf(path, pages=2)
            before = path.stat().st_size
            changes = self.convert(path)
            self.assertEqual((changes.converted, changes.pages), (1, {0, 1}))
            self.assertEqual(fontfile_kind(path), ["/FontFile3"])
            with open_pdf(path) as pdf:
                program_ = pdf.pages[0].Resources.Font.F1.FontDescriptor.FontFile3
                self.assertEqual(program_.Subtype, "/Type1C")
            self.assertLess(path.stat().st_size, before)

    def test_skipped_and_unclear_programs_stay(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "t1.pdf"
            make_type1_pdf(path)
            with open_pdf(path) as pdf:
                key = pdf.pages[0].Resources.Font.F1.FontDescriptor.FontFile.objgen
            self.assertEqual(self.convert(path, frozenset({key})).converted, 0)
            make_type1_pdf(path, orphan_user=True)
            self.assertEqual(self.convert(path).converted, 0)
            self.assertEqual(fontfile_kind(path), ["/FontFile"])

    def test_an_interactive_form_does_not_stop_the_conversion(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "t1.pdf"
            make_type1_pdf(path)
            with pikepdf.open(path, allow_overwriting_input=True) as pdf:
                pdf.Root.AcroForm = pikepdf.Dictionary(Fields=[pdf.make_indirect(
                    pikepdf.Dictionary(FT=pikepdf.Name.Tx, T=pikepdf.String("x")))])
                pdf.save(path)
            self.assertEqual(self.convert(path).converted, 1)


@unittest.skipUnless(need("pdftoppm"), "pdftoppm is not installed")
class RenderingTests(unittest.TestCase):
    def test_converted_pages_render_pixel_for_pixel_the_same(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "t1.pdf", Path(root) / "out.pdf"
            make_type1_pdf(source, pages=2)
            report = lossless.build(source, target, Path(root))
            self.assertIn("type1→cff×1", report.notes)
            self.assertEqual(report.visual_pages, {0, 1})
            self.assertTrue(identical_pages(source, target, [1, 2], Path(root) / "x"))

    def test_optimize_keeps_the_conversion(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "t1.pdf"
            make_type1_pdf(path)
            status, saved = optimize_pdf(path, jobs=2)
            self.assertEqual(status, "compressed (lossless)")
            self.assertGreater(saved, 0)
            self.assertEqual(fontfile_kind(path), ["/FontFile3"])


if __name__ == "__main__":
    unittest.main()
