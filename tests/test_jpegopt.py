from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pikepdf
from pikepdf import Dictionary, Name, Pdf
from PIL import Image

from pdfopt import jpegopt
from pdfopt.pdfmodel import open_pdf
from pdfopt.ssim import ssim

from .helpers import _page, image_stream, jpeg_bytes, make_flate_image_pdf, make_scan_pdf


def photo(width=900, height=700, seed=2) -> Image.Image:
    """Smooth gradients plus fine noise: JPEG-friendly but not trivial."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack([128 + 100 * np.sin(x / 90), 128 + 100 * np.cos(y / 70),
                     128 + 90 * np.sin((x + y) / 130)], axis=2)
    return Image.fromarray((base + rng.normal(0, 1.5, base.shape)).clip(0, 255).astype(np.uint8))


def photo_pdf(path: Path, *, quality=97, size=(900, 700), flate=False, dpi=None):
    image = photo(*size)
    pdf = Pdf.new()
    if flate:
        import zlib
        stream = image_stream(pdf, zlib.compress(image.tobytes(), 6), *size,
                              filter_name=Name.FlateDecode)
    else:
        stream = image_stream(pdf, jpeg_bytes(image, quality), *size,
                              filter_name=Name.DCTDecode)
    width_pt = size[0] * 72 / dpi if dpi else 300
    height_pt = size[1] * 72 / dpi if dpi else 300 * size[1] / size[0]
    _page(pdf, width_pt, height_pt,
          f"q {width_pt} 0 0 {height_pt} 0 0 cm /Im0 Do Q".encode(),
          Dictionary(XObject=Dictionary(Im0=stream)))
    pdf.save(path)


class BisectTests(unittest.TestCase):
    def test_finds_the_threshold(self):
        calls = []

        def passes(quality):
            calls.append(quality)
            return b"x" * quality if quality >= 61 else None

        quality, _data = jpegopt.bisect_quality(passes)
        self.assertLessEqual(abs(quality - 61), 2)
        self.assertLessEqual(len(calls), 8)

    def test_nothing_passes(self):
        self.assertIsNone(jpegopt.bisect_quality(lambda quality: None))

    def test_everything_passes(self):
        quality, _ = jpegopt.bisect_quality(lambda quality: b"ok")
        self.assertLessEqual(quality, jpegopt.QUALITY_RANGE[0] + 2)


class GeometryTests(unittest.TestCase):
    def test_effective_dpi_follows_the_matrix(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "a.pdf"
            make_scan_pdf(path, text_page=False)   # 400 px over 360 pt = 80 dpi
            with open_pdf(path) as pdf:
                dpis = jpegopt.effective_dpis(pdf)
            self.assertEqual(len(dpis), 1)
            self.assertAlmostEqual(next(iter(dpis.values())), 400 * 72 / 360, delta=0.5)

    def test_nested_q_and_scaling_are_tracked(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "a.pdf"
            pdf = Pdf.new()
            stream = image_stream(pdf, jpeg_bytes(photo(200, 200)), 200, 200,
                                  filter_name=Name.DCTDecode)
            _page(pdf, 100, 100, b"q 0.5 0 0 0.5 0 0 cm q 144 0 0 144 0 0 cm /Im0 Do Q Q",
                  Dictionary(XObject=Dictionary(Im0=stream)))
            pdf.save(path)
            with open_pdf(path) as opened:
                dpi = next(iter(jpegopt.effective_dpis(opened).values()))
            self.assertAlmostEqual(dpi, 200 * 72 / 72, delta=0.5)


class ReencodeTests(unittest.TestCase):
    def test_result_meets_the_threshold_at_view_size(self):
        bitmap = photo()
        stored = len(jpeg_bytes(bitmap, 97))
        with tempfile.TemporaryDirectory() as root:
            found = jpegopt.reencode(bitmap, stored, None, 0.93, Path(root))
        self.assertIsNotNone(found)
        data, size = found
        self.assertLess(len(data), stored * jpegopt.MIN_GAIN)
        with Image.open(BytesIO(data)) as decoded:
            self.assertEqual(decoded.size, size)
            score = ssim(np.asarray(bitmap),
                         np.asarray(decoded.convert("RGB").resize(bitmap.size)))
        self.assertGreater(score, 0.9)

    def test_high_resolution_is_downsampled(self):
        bitmap = photo(1800, 1400)
        stored = len(jpeg_bytes(bitmap, 97))
        with tempfile.TemporaryDirectory() as root:
            found = jpegopt.reencode(bitmap, stored, 600.0, 0.93, Path(root))
        self.assertIsNotNone(found)
        self.assertLess(found[1][0], bitmap.width)

    def test_tiny_gain_is_rejected(self):
        bitmap = photo(300, 300)
        stored = len(jpeg_bytes(bitmap, 30))
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(jpegopt.reencode(bitmap, stored, None, 0.99, Path(root)))

    def test_tighten_keeps_pixels_and_never_grows(self):
        plain = jpegopt.encode(photo(), 80)
        tight = jpegopt.tighten(plain)
        self.assertLess(len(tight), len(plain))
        with Image.open(BytesIO(plain)) as a, Image.open(BytesIO(tight)) as b:
            self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(jpegopt.tighten(b"not a jpeg"), b"not a jpeg")


class DocumentTests(unittest.TestCase):
    def run_stage(self, path, minimum=0.95):
        with tempfile.TemporaryDirectory() as root, open_pdf(path) as pdf:
            counts, changes = jpegopt.optimize_jpegs(pdf, Path(root), minimum, 2)
            target = Path(root) / "out.pdf"
            pdf.save(target)
            touched = {page + 1 for change in changes for page in change.pages}
            return counts, touched, target.stat().st_size

    def test_high_quality_jpeg_shrinks(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "p.pdf"
            photo_pdf(path)
            counts, touched, size = self.run_stage(path)
            self.assertEqual(touched, {1})
            self.assertTrue(counts)
            self.assertLess(size, path.stat().st_size * 0.8)

    def test_photographic_flate_becomes_jpeg(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "p.pdf"
            photo_pdf(path, flate=True)
            _, touched, size = self.run_stage(path)
            self.assertEqual(touched, {1})
            self.assertLess(size, path.stat().st_size * 0.3)

    def test_flat_graphics_stay_flate(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "g.pdf"
            flat = np.zeros((400, 400, 3), np.uint8)
            flat[100:300, 100:300] = (200, 30, 30)
            make_flate_image_pdf(path, flat)
            _, touched, _ = self.run_stage(path)
            self.assertEqual(touched, set())

    def test_images_with_masks_or_decode_are_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "p.pdf"
            photo_pdf(path)
            with pikepdf.open(path, allow_overwriting_input=True) as pdf:
                pdf.pages[0].Resources.XObject.Im0.Decode = pikepdf.Array([1, 0, 1, 0, 1, 0])
                pdf.save(path)
            _, touched, _ = self.run_stage(path)
            self.assertEqual(touched, set())

    def test_revert_restores_the_original_stream(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "p.pdf"
            photo_pdf(path, size=(1800, 1400), dpi=600)
            with tempfile.TemporaryDirectory() as scratch, open_pdf(path) as pdf:
                obj = pdf.pages[0].Resources.XObject.Im0
                before = (bytes(obj.read_raw_bytes()), int(obj.Width), int(obj.Height))
                _, changes = jpegopt.optimize_jpegs(pdf, Path(scratch), 0.93, 2)
                self.assertEqual(len(changes), 1)
                self.assertNotEqual(bytes(obj.read_raw_bytes()), before[0])
                self.assertEqual(changes[0].pages, {0})
                jpegopt.revert(changes[0])
                self.assertEqual((bytes(obj.read_raw_bytes()), int(obj.Width),
                                  int(obj.Height)), before)
                self.assertEqual(str(obj.Filter), "/DCTDecode")

    def test_low_resolution_images_are_judged_at_render_size(self):
        self.assertAlmostEqual(jpegopt.view_factor(700, 900, 72.0), 200 / 72)
        self.assertEqual(jpegopt.view_factor(700, 900, 20.0), jpegopt.MAX_UPSCALE)
        self.assertAlmostEqual(jpegopt.view_factor(1000, 1000, 100.0), 2.0)
        self.assertLess(jpegopt.view_factor(5000, 5000, 100.0), 2.0)   # pixel cap
        self.assertEqual(jpegopt.view_factor(1000, 1000, None), 1.0)

    def test_written_image_is_a_valid_jpeg_with_new_size(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "p.pdf"
            photo_pdf(path, size=(1800, 1400), dpi=600)
            with tempfile.TemporaryDirectory() as scratch, open_pdf(path) as pdf:
                jpegopt.optimize_jpegs(pdf, Path(scratch), 0.93, 2)
                obj = pdf.pages[0].Resources.XObject.Im0
                with Image.open(BytesIO(bytes(obj.read_raw_bytes()))) as decoded:
                    self.assertEqual(decoded.size, (int(obj.Width), int(obj.Height)))
                self.assertEqual(str(obj.Filter), "/DCTDecode")


if __name__ == "__main__":
    unittest.main()
