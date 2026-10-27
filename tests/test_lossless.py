from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
import pikepdf
from pikepdf import Dictionary, Pdf
from PIL import Image

from pdfopt import imagedata, lossless
from pdfopt.pdfmodel import open_pdf
from pdfopt.render import identical_pages

from .helpers import (
    bilevel_bitmap,
    g4_stream,
    jpeg_bytes,
    make_bilevel_pdf,
    make_flate_image_pdf,
    make_scan_pdf,
    make_text_pdf,
    need,
    stripes,
    text_image,
)


def build(source: Path, scratch: Path, **options):
    target = scratch / "out.pdf"
    report = lossless.build(source, target, scratch, jobs=2, **options)
    return target, report


def first_image(path: Path):
    with open_pdf(path) as pdf:
        page = pdf.pages[0]
        obj = page.Resources.XObject.Im0
        raw = bytes(obj.read_raw_bytes())
        try:
            decoded = obj.read_bytes()
        except pikepdf.PdfError:   # DCT/JBIG2/CCITT are not decoded by qpdf
            decoded = raw
        return dict(obj.items()), raw, decoded


class StructureTests(unittest.TestCase):
    def test_text_pdf_shrinks_and_content_is_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "book.pdf"
            make_text_pdf(source)
            target, _ = build(source, Path(root))
            self.assertLess(target.stat().st_size, source.stat().st_size)
            with open_pdf(source) as a, open_pdf(target) as b:
                self.assertEqual(a.pages[0].Contents.read_bytes(),
                                 b.pages[0].Contents.read_bytes())

    def test_object_streams_are_written(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "book.pdf"
            make_text_pdf(source)
            target, _ = build(source, Path(root), ect=False)
            self.assertIn(b"/ObjStm", target.read_bytes())

    def test_thumbnails_are_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "book.pdf"
            make_text_pdf(source)
            with pikepdf.open(source, allow_overwriting_input=True) as pdf:
                pdf.pages[0].obj.Thumb = pdf.make_stream(b"x" * 4000)
                pdf.save(source)
            target, report = build(source, Path(root))
            self.assertIn("thumbs", report.notes)
            with open_pdf(target) as pdf:
                self.assertNotIn("/Thumb", pdf.pages[0].obj)

    def test_identical_streams_are_merged(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "dup.pdf"
            pdf = Pdf.new()
            payload = np.random.default_rng(1).integers(0, 255, 5000, dtype=np.uint8).tobytes()
            for _ in range(3):
                page = pdf.add_blank_page(page_size=(100, 100))
                page.Resources = Dictionary(XObject=Dictionary(
                    Im0=pdf.make_stream(payload)))
            pdf.save(source)
            target, report = build(source, Path(root), ect=False)
            self.assertIn("dedupe", report.notes)
            self.assertLess(target.stat().st_size, source.stat().st_size * 0.6)
            with open_pdf(target) as pdf:
                ids = {page.Resources.XObject.Im0.objgen for page in pdf.pages}
            self.assertEqual(len(ids), 1)

    def test_ect_never_hurts_and_streams_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "book.pdf"
            make_text_pdf(source)
            plain, _ = build(source, Path(root), ect=False)
            small = Path(root) / "z"
            small.mkdir()
            packed, _ = build(source, small, ect=True)
            self.assertLessEqual(packed.stat().st_size, plain.stat().st_size)


@unittest.skipUnless(lossless.ect_available(), "модуль ECT не собран")
class EctTests(unittest.TestCase):
    def test_zlib_wrapper_decodes_and_beats_zlib(self):
        data = (b"pdfopt " * 500 + bytes(range(256)) * 20) * 3
        packed = lossless.deflate_zlib(data)
        self.assertEqual(zlib.decompress(packed), data)
        self.assertLessEqual(len(packed), len(zlib.compress(data, 9)))

    def test_empty_input(self):
        self.assertEqual(lossless.deflate_zlib(b""), zlib.compress(b"", 9)[:0] +
                         b"\x78\xda\x03\x00\x00\x00\x00\x01")

    def test_modes_outside_2_to_9_are_refused(self):
        # Mode 1 never returned (on a 2 MB image band); it must not be reachable.
        for mode in (0, 1, 10):
            with self.subTest(mode), self.assertRaises(ValueError):
                lossless._ectdeflate.deflate(b"pdfopt" * 100, mode)
        for mode in (2, 6, 9):
            data = b"pdfopt" * 100
            self.assertEqual(zlib.decompress(lossless.deflate_zlib(data, mode)), data)

    def test_big_streams_get_the_faster_level(self):
        self.assertEqual(lossless.ect_mode(lossless.ECT_FAST_ABOVE), 9)
        self.assertEqual(lossless.ect_mode(lossless.ECT_FAST_ABOVE + 1), 6)
        pdf = Pdf.new()
        small, big = (pdf.make_stream(bytes(size)) for size in (1000, 5000))
        modes = []
        real = lossless.deflate_zlib

        def spy(data, mode=9):
            modes.append((len(data), mode))
            return real(data, mode)

        with patch.object(lossless, "ECT_FAST_ABOVE", 2000), \
                patch.object(lossless, "deflate_zlib", spy):
            lossless.ect_streams(pdf, 1)
        self.assertEqual(sorted(modes), [(1000, 9), (5000, 6)])
        self.assertEqual(small.read_bytes(), bytes(1000))
        self.assertEqual(big.read_bytes(), bytes(5000))

    def test_streams_beyond_the_budget_are_not_inflated(self):
        pdf = Pdf.new()
        stream = pikepdf.Stream(pdf, b"")
        stream.write(zlib.compress(bytes(64 << 20), 9), filter=pikepdf.Name.FlateDecode)
        pdf.make_indirect(stream)
        before = bytes(stream.read_raw_bytes())
        with patch.object(lossless, "ECT_BUDGET", 1 << 20):
            lossless.ect_streams(pdf, 1)
        self.assertEqual(bytes(stream.read_raw_bytes()), before)


class ImageTests(unittest.TestCase):
    def test_gray_rgb_becomes_devicegray_with_same_pixels(self):
        gray = stripes(gray=True)
        rgb = np.stack([gray] * 3, axis=2)
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "rgb.pdf"
            make_flate_image_pdf(source, rgb)
            target, _report = build(source, Path(root))
            entries, _, data = first_image(target)
            self.assertEqual(entries["/ColorSpace"], "/DeviceGray")
            self.assertEqual(data, gray.tobytes())
            self.assertLess(target.stat().st_size, source.stat().st_size)

    def test_black_and_white_gray_goes_one_bit(self):
        gray = (np.random.default_rng(3).random((300, 300)) > 0.5).astype(np.uint8) * 255
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "bw.pdf"
            make_flate_image_pdf(source, gray)
            target, _ = build(source, Path(root), jbig2=False)
            entries, _, data = first_image(target)
            self.assertEqual(entries["/BitsPerComponent"], 1)
            unpacked = np.unpackbits(np.frombuffer(data, np.uint8).reshape(300, -1),
                                     axis=1)[:, :300] * 255
            self.assertTrue(np.array_equal(unpacked, gray))

    def test_picture_with_other_values_keeps_eight_bits(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "g.pdf"
            make_flate_image_pdf(source, stripes(gray=True))
            target, _ = build(source, Path(root))
            self.assertEqual(first_image(target)[0]["/BitsPerComponent"], 8)

    def test_gray_rgb_with_a_decode_array_stays_rgb(self):
        # Six /Decode values would be invalid for a DeviceGray image.
        rgb = np.stack([stripes(gray=True)] * 3, axis=2)
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "rgb.pdf"
            make_flate_image_pdf(source, rgb, Decode=pikepdf.Array([1, 0, 1, 0, 1, 0]))
            target, _ = build(source, Path(root))
            entries = first_image(target)[0]
            self.assertEqual(entries["/ColorSpace"], "/DeviceRGB")
            self.assertEqual(len(entries["/Decode"]), 6)

    def test_flate_bomb_is_inflated_only_as_far_as_the_image_needs(self):
        pdf = Pdf.new()
        bomb = zlib.compress(bytes(256 << 20), 9)        # 256 MB of zeros, ~250 KB
        obj = pikepdf.Stream(pdf, b"")
        obj.write(bomb, filter=pikepdf.Name.FlateDecode)
        rows = imagedata.raw_samples(obj, 10, 10, 1, 8)
        self.assertEqual(rows.shape, (10, 10))
        # A declared size beyond the limit is not decoded at all.
        self.assertIsNone(imagedata.raw_samples(obj, 1 << 16, 1 << 14, 1, 8))

    def test_color_key_masked_images_are_left_alone(self):
        gray = np.stack([stripes(gray=True)] * 3, axis=2)
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "ck.pdf"
            make_flate_image_pdf(source, gray, Mask=pikepdf.Array([0, 0, 0, 0, 0, 0]))
            target, _ = build(source, Path(root))
            self.assertEqual(first_image(target)[0]["/ColorSpace"], "/DeviceRGB")

    def test_predictor_keeps_decoded_samples(self):
        picture = stripes()
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "p.pdf"
            make_flate_image_pdf(source, picture)
            target, _report = build(source, Path(root))
            _entries, raw, data = first_image(target)
            self.assertEqual(data, picture.tobytes())
            self.assertLess(len(raw), 0.5 * len(picture.tobytes()))

    def test_png_predict_is_invertible(self):
        rows = np.random.default_rng(5).integers(0, 255, (40, 33), dtype=np.uint8)
        encoded = np.frombuffer(imagedata.png_predict(rows, 3), np.uint8).reshape(40, 34)
        decoded = np.zeros_like(rows)
        bpp = 3
        for y in range(40):
            kind, line = encoded[y, 0], encoded[y, 1:].astype(np.int32)
            above = decoded[y - 1].astype(np.int32) if y else np.zeros(33, np.int32)
            out = np.zeros(33, np.int32)
            for x in range(33):
                left = out[x - bpp] if x >= bpp else 0
                upleft = above[x - bpp] if x >= bpp else 0
                if kind == 0:
                    pred = 0
                elif kind == 1:
                    pred = left
                elif kind == 2:
                    pred = above[x]
                elif kind == 3:
                    pred = (left + above[x]) // 2
                else:
                    p = left + above[x] - upleft
                    pa, pb, pc = abs(p - left), abs(p - above[x]), abs(p - upleft)
                    pred = left if pa <= pb and pa <= pc else above[x] if pb <= pc else upleft
                out[x] = (line[x] + pred) & 255
            decoded[y] = out
        self.assertTrue(np.array_equal(decoded, rows))

    def test_g4_decoder_matches_pillow(self):
        bitmap = bilevel_bitmap(203, 57)
        rows = imagedata.g4_bitmap(g4_stream(bitmap), 203, 57, False)
        # Padding bits are not significant.
        expected = np.frombuffer(bitmap.tobytes(), np.uint8).reshape(57, -1)
        self.assertTrue(np.array_equal(rows >> 5, expected >> 5))
        inverted = imagedata.g4_bitmap(g4_stream(bitmap), 203, 57, True)
        self.assertTrue(np.array_equal(inverted, ~rows))


@unittest.skipUnless(need("jbig2", "pdftoppm"), "нужны jbig2 и pdftoppm")
class Jbig2Tests(unittest.TestCase):
    def check(self, kind, mask):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "bw.pdf"
            make_bilevel_pdf(source, kind=kind, mask=mask)
            target, _report = build(source, Path(root))
            self.assertLess(target.stat().st_size, source.stat().st_size)
            with open_pdf(target) as pdf:
                obj = pdf.pages[0].Resources.XObject.Im0
                self.assertEqual(str(obj.Filter), "/JBIG2Decode")
            self.assertTrue(identical_pages(source, target, [1], Path(root)))

    def test_flate_gray_page_to_jbig2(self):
        self.check("flate", False)

    def test_flate_image_mask_to_jbig2(self):
        self.check("flate", True)

    def test_group4_page_to_jbig2(self):
        self.check("g4", False)

    def test_group4_image_mask_to_jbig2(self):
        self.check("g4", True)


class JpegTests(unittest.TestCase):
    def test_pixels_survive_and_file_shrinks(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source, optimize=False)
            target, report = build(source, Path(root))
            _, _, before = first_image(source)
            _, _, after = first_image(target)
            self.assertIn("jpeg×1", report.notes)
            with Image.open(__import__("io").BytesIO(before)) as a, \
                    Image.open(__import__("io").BytesIO(after)) as b:
                self.assertEqual(a.tobytes(), b.tobytes())
            self.assertLess(target.stat().st_size, source.stat().st_size)

    def test_mozjpeg_rewrite_keeps_every_pixel(self):
        original = jpeg_bytes(text_image(400, 300, color=True), quality=90)
        smaller = lossless.optimize_jpeg(original)
        self.assertIsNotNone(smaller)
        self.assertLess(len(smaller), len(original))
        self.assertTrue(lossless._same_pixels(original, smaller))
        self.assertIsNone(lossless.optimize_jpeg(b"not a jpeg"))
        # Already optimal: nothing to gain, nothing returned.
        self.assertIsNone(lossless.optimize_jpeg(smaller))

    def test_same_pixels_detects_difference(self):
        one = jpeg_bytes(text_image(64, 64))
        other = jpeg_bytes(text_image(64, 64, paper=(200, 30, 30)))
        self.assertTrue(lossless._same_pixels(one, one))
        self.assertFalse(lossless._same_pixels(one, other))
        self.assertFalse(lossless._same_pixels(b"no", b"no"))

    def test_dct_with_decode_parms_is_left_alone(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source, optimize=False)
            with pikepdf.open(source, allow_overwriting_input=True) as pdf:
                pdf.pages[0].Resources.XObject.Im0.DecodeParms = Dictionary(ColorTransform=1)
                pdf.save(source)
            _, report = build(source, Path(root))
            self.assertNotIn("jpeg×1", report.notes)


@unittest.skipUnless(need("pdftoppm"), "pdftoppm не установлен")
class VisualExactnessTests(unittest.TestCase):
    def test_changed_pages_render_identically(self):
        gray = stripes(gray=True)
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "rgb.pdf"
            make_flate_image_pdf(source, np.stack([gray] * 3, axis=2))
            target, report = build(source, Path(root))
            self.assertEqual(report.visual_pages, {0})
            self.assertTrue(identical_pages(source, target, [1], Path(root)))

    def test_exactness_check_notices_real_differences(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = Path(root) / "a.pdf", Path(root) / "b.pdf"
            make_flate_image_pdf(first, stripes())
            make_flate_image_pdf(second, stripes()[::-1].copy())
            self.assertFalse(identical_pages(first, second, [1], Path(root)))


if __name__ == "__main__":
    unittest.main()
