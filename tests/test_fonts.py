from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fontTools.ttLib import TTFont
import pikepdf

from pdfopt import fonts, lossless
from pdfopt.pdfmodel import open_pdf
from pdfopt.render import identical_pages

from .helpers import make_font_pdf, need


def program(path: Path) -> TTFont:
    with open_pdf(path) as pdf:
        for obj in pdf.objects:
            if isinstance(obj, pikepdf.Dictionary) and "/FontFile2" in obj:
                return TTFont(BytesIO(bytes(obj.FontFile2.read_bytes())))
    raise AssertionError("no embedded TrueType font")


def outlines(tt: TTFont) -> set[str]:
    """Glyphs that still have contours."""
    glyf = tt["glyf"]
    return {name for name in tt.getGlyphOrder() if glyf[name].numberOfContours}


def cut(path: Path) -> tuple[set[int], int]:
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        result = fonts.subset_fonts(pdf)
        pdf.save(path)
    return result


class SubsetTests(unittest.TestCase):
    def test_simple_truetype_keeps_only_the_shown_glyphs(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "f.pdf"
            make_font_pdf(path, kind="simple", text="ABAC")
            before = path.stat().st_size
            pages, count = cut(path)
            self.assertEqual((pages, count), ({0}, 1))
            tt = program(path)
            # Glyph ids are kept (content streams do not change); the order
            # ends with the last glyph in use.
            self.assertEqual(tt.getGlyphOrder()[:4], [".notdef", "g000", "g001", "g002"])
            self.assertEqual(outlines(tt), {".notdef", "g000", "g001", "g002"})
            self.assertLess(path.stat().st_size, before / 2)

    def test_cid_font_keeps_the_glyph_ids_of_its_codes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "f.pdf"
            make_font_pdf(path, kind="cid", text="ABAC")
            self.assertEqual(cut(path)[1], 1)
            self.assertEqual(outlines(program(path)), {".notdef", "g000", "g001", "g002"})

    def test_text_in_forms_and_annotation_appearances_counts(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "f.pdf"
            make_font_pdf(path, kind="simple", text="AB", in_form=True, annotation=True)
            cut(path)
            # "Z" (g025) is shown only by the annotation.
            self.assertEqual(outlines(program(path)), {".notdef", "g000", "g001", "g025"})

    def test_mac_only_cmap_survives_the_cut(self):
        # LibreOffice fonts: no /Encoding, one (1, 0) cmap table.
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "f.pdf"
            make_font_pdf(path, kind="simple", text="ABAC", mac_only=True)
            before = path.stat().st_size
            self.assertEqual(cut(path)[1], 1)
            tt = program(path)
            self.assertEqual([(t.platformID, t.platEncID) for t in tt["cmap"].tables], [(1, 0)])
            self.assertEqual(sorted(tt["cmap"].tables[0].cmap), [0x41, 0x42, 0x43])
            self.assertLess(path.stat().st_size, before / 2)

    def test_fonts_are_left_alone_when_use_is_unclear(self):
        cases = {"acroform": {"acroform": True},
                 "orphan user": {"orphan_user": True},
                 "non-identity cmap": {"kind": "cid", "encoding": "/UniGB-UCS2-H"}}
        for label, options in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "f.pdf"
                make_font_pdf(path, **options)
                self.assertEqual(cut(path), (set(), 0))
                self.assertEqual(len(outlines(program(path))), 201)

    def test_unparsable_content_disables_cutting(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "f.pdf"
            make_font_pdf(path)
            with patch.object(pikepdf, "parse_content_stream",
                              side_effect=pikepdf.PdfError("damaged")):
                self.assertEqual(cut(path), (set(), 0))


@unittest.skipUnless(need("pdftoppm"), "pdftoppm не установлен")
class LosslessBuildTests(unittest.TestCase):
    def test_build_cuts_fonts_and_renders_the_same(self):
        for kind, mac_only in (("simple", False), ("cid", False), ("simple", True)):
            with self.subTest(kind, mac_only=mac_only), tempfile.TemporaryDirectory() as root:
                scratch = Path(root)
                source = scratch / "f.pdf"
                make_font_pdf(source, kind=kind, annotation=True, mac_only=mac_only)
                target = scratch / "out.pdf"
                report = lossless.build(source, target, scratch, jobs=1)
                self.assertIn("fonts×1", report.notes)
                self.assertEqual(report.visual_pages, {0})
                self.assertLess(target.stat().st_size, source.stat().st_size / 2)
                self.assertTrue(identical_pages(source, target, [1], scratch / "x"))

    def test_safe_build_leaves_fonts_alone(self):
        with tempfile.TemporaryDirectory() as root:
            scratch = Path(root)
            source = scratch / "f.pdf"
            make_font_pdf(source)
            report = lossless.build(source, scratch / "out.pdf", scratch, jobs=1,
                                    transforms=False)
            self.assertFalse(any(note.startswith("fonts") for note in report.notes))


if __name__ == "__main__":
    unittest.main()
