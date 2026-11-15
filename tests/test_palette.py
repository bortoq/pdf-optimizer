"""Images as indices into a colour table, and the packed xref streams."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
import pikepdf
from PIL import Image, ImageDraw, ImageFont

from pdfopt import imagedata, jpegopt, lossless
from pdfopt.pdfmodel import open_pdf
from pdfopt.render import identical_pages
from pdfopt.ssim import ssim

from .helpers import make_flate_image_pdf, make_text_pdf, need
from .test_jpegopt import photo


def screenshot(width=900, height=700, *, gray=False) -> Image.Image:
    """Anti-aliased text and flat boxes: many levels, but most pixels repeat."""
    image = Image.new("RGB", (width, height), (246, 246, 240))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=26)
    for row in range(0, height - 40, 44):
        colour = ((20, 20, 20), (30, 60, 160), (170, 30, 30))[row // 44 % 3]
        draw.text((24, row + 8), "Window title - settings 0123456789 pdfopt", font=font,
                  fill=colour)
    draw.rectangle((600, 100, 860, 300), fill=(60, 140, 90))
    return image.convert("L") if gray else image


def few_colours(width=300, height=200) -> np.ndarray:
    """An RGB picture of five colours in bands."""
    colours = np.array([(250, 250, 250), (200, 30, 30), (30, 30, 200), (30, 160, 60),
                        (10, 10, 10)], np.uint8)
    band = (np.arange(width)[None, :] // 7 + np.arange(height)[:, None] // 11) % 5
    return colours[band]


def image_of(path: Path) -> pikepdf.Object:
    with open_pdf(path) as pdf:
        image = pdf.pages[0].Resources.XObject.Im0
        return pikepdf.Dictionary({key: value for key, value in image.items()
                                   if key != "/Length"})


class HelperTests(unittest.TestCase):
    def test_indices_are_packed_most_significant_first(self):
        indices = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 1]], np.uint8)
        self.assertEqual(imagedata.pack_indices(indices, 1).tolist(), [[0b10110001, 0b10000000]])
        self.assertEqual(imagedata.pack_indices(np.array([[3, 0, 2]], np.uint8), 2).tolist(),
                         [[0b11001000]])
        self.assertEqual(imagedata.pack_indices(np.array([[15, 1, 7]], np.uint8), 4).tolist(),
                         [[0xF1, 0x70]])
        self.assertEqual([imagedata.index_bits(n) for n in (2, 3, 4, 5, 16, 17, 256)],
                         [1, 2, 2, 4, 4, 8, 8])

    def test_exact_palette_rebuilds_every_pixel(self):
        pixels = few_colours()
        indices, table = imagedata.exact_palette(pixels)
        self.assertEqual(len(table), 5)
        self.assertTrue(np.array_equal(table[indices], pixels))
        self.assertIsNone(imagedata.exact_palette(pixels, limit=4))


@unittest.skipUnless(need("pdftoppm"), "pdftoppm is not installed")
class LosslessPaletteTests(unittest.TestCase):
    def build(self, array, **options):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "s.pdf", Path(root) / "t.pdf"
            make_flate_image_pdf(source, array, **options)
            report = lossless.build(source, target, Path(root))
            same = identical_pages(source, target, [1], Path(root) / "x")
            return report, image_of(target), same, source.stat().st_size, target.stat().st_size

    def test_rgb_with_few_colours_becomes_indexed(self):
        report, image, same, before, after = self.build(few_colours())
        self.assertTrue(same)
        self.assertIn("palette×1", report.notes)
        self.assertEqual(str(image.ColorSpace[0]), "/Indexed")
        self.assertEqual(str(image.ColorSpace[1]), "/DeviceRGB")
        self.assertEqual((int(image.ColorSpace[2]), int(image.BitsPerComponent)), (4, 4))
        self.assertLess(after, before)

    def test_gray_with_few_levels_gets_narrow_indices(self):
        levels = np.array([0, 90, 180, 255], np.uint8)
        gray = levels[np.add.outer(np.arange(300) // 9, np.arange(400) // 13) % 4]
        _, image, same, _, _ = self.build(gray)
        self.assertTrue(same)
        self.assertEqual(str(image.ColorSpace[1]), "/DeviceGray")
        self.assertEqual(int(image.BitsPerComponent), 2)

    def test_gray_rgb_with_few_levels_becomes_indexed_gray(self):
        levels = np.array([0, 120, 255], np.uint8)
        gray = levels[np.add.outer(np.arange(300) // 9, np.arange(400) // 13) % 3]
        report, image, same, _, _ = self.build(np.stack([gray] * 3, axis=2))
        self.assertTrue(same)
        self.assertIn("gray+palette×1", report.notes)
        self.assertEqual(str(image.ColorSpace[1]), "/DeviceGray")

    def test_many_levels_and_masks_stay_as_they_are(self):
        smooth = (np.add.outer(np.arange(300), np.arange(400)) % 256).astype(np.uint8)
        report, image, _, _, _ = self.build(smooth)
        self.assertEqual(str(image.ColorSpace), "/DeviceGray")
        self.assertFalse(any("palette" in note for note in report.notes))
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "s.pdf", Path(root) / "t.pdf"
            make_flate_image_pdf(source, few_colours())
            mask = (np.arange(300)[None, :] // 50 % 2 * 255).repeat(200, 0).astype(np.uint8)
            with pikepdf.open(source, allow_overwriting_input=True) as pdf:
                image = pdf.pages[0].Resources.XObject.Im0
                image.SMask = pdf.make_stream(
                    zlib.compress(mask.tobytes(), 1), Type=pikepdf.Name.XObject,
                    Subtype=pikepdf.Name.Image, Width=300, Height=200,
                    ColorSpace=pikepdf.Name.DeviceGray, BitsPerComponent=8)
                pdf.save(source)
            lossless.build(source, target, Path(root))
            with open_pdf(target) as pdf:
                # A soft mask must stay DeviceGray, whatever it holds.
                self.assertEqual(str(pdf.pages[0].Resources.XObject.Im0.SMask.ColorSpace),
                                 "/DeviceGray")
            self.assertTrue(identical_pages(source, target, [1], Path(root) / "x"))


class PhotographicTests(unittest.TestCase):
    def rows(self, image: Image.Image) -> np.ndarray:
        array = np.asarray(image)
        return array.reshape(array.shape[0], -1)

    def test_gray_photos_and_gray_screenshots_are_told_apart(self):
        self.assertTrue(jpegopt._photographic(self.rows(photo().convert("L")), 1))
        self.assertFalse(jpegopt._photographic(self.rows(screenshot(gray=True)), 1))
        posterized = (np.asarray(photo().convert("L")) // 8 * 8).astype(np.uint8)
        self.assertFalse(jpegopt._photographic(posterized, 1))     # 32 levels
        self.assertTrue(jpegopt._photographic(self.rows(photo()), 3))
        self.assertFalse(jpegopt._photographic(self.rows(screenshot()), 3))

    def test_requantize_keeps_the_ssim_and_shrinks(self):
        for gray in (False, True):
            with self.subTest(gray=gray):
                image = screenshot(gray=gray)
                stored = len(zlib.compress(image.tobytes(), 9))
                found = jpegopt.requantize(image, stored, 300, 0.9)
                self.assertIsNotNone(found)
                self.assertLess(len(found.data) + len(found.table), stored * 0.9)
                # Decoded the way a PDF reader decodes it.
                pdf = pikepdf.new()
                stream = pdf.make_stream(b"", Type=pikepdf.Name.XObject,
                                         Subtype=pikepdf.Name.Image, Width=image.width,
                                         Height=image.height, BitsPerComponent=8,
                                         ColorSpace=pikepdf.Name.DeviceGray if gray
                                         else pikepdf.Name.DeviceRGB)
                imagedata.write_indexed(stream, found.data, found.parms, found.bits,
                                        found.table, found.count)
                shown = pikepdf.PdfImage(stream).as_pil_image().convert(image.mode)
                self.assertGreaterEqual(ssim(np.asarray(image), np.asarray(shown)), 0.9)


class LossyDocumentTests(unittest.TestCase):
    def run_stage(self, image: Image.Image, minimum=0.9):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "p.pdf"
            make_flate_image_pdf(path, np.asarray(image))
            with open_pdf(path) as pdf:
                counts, _ = jpegopt.optimize_jpegs(pdf, Path(root), minimum, 2)
                target = Path(root) / "out.pdf"
                pdf.save(target)
            with open_pdf(target) as pdf:
                image_ = pdf.pages[0].Resources.XObject.Im0
                return counts, str(image_.get("/Filter")), image_.get("/ColorSpace"), \
                    path.stat().st_size, target.stat().st_size

    def test_a_gray_photo_becomes_a_jpeg(self):
        _, filter_name, _, before, after = self.run_stage(photo().convert("L"))
        self.assertEqual(filter_name, "/DCTDecode")
        self.assertLess(after, before * 0.5)

    def test_a_screenshot_gets_a_palette_and_keeps_its_space(self):
        for gray in (False, True):
            with self.subTest(gray=gray):
                counts, filter_name, space, before, after = self.run_stage(
                    screenshot(gray=gray))
                self.assertEqual(counts["palette"], 1)
                self.assertEqual(filter_name, "/FlateDecode")
                self.assertEqual(str(space[0]), "/Indexed")
                self.assertEqual(str(space[1]), "/DeviceGray" if gray else "/DeviceRGB")
                self.assertLess(after, before)

    def test_a_palette_can_be_reverted(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "p.pdf"
            make_flate_image_pdf(path, np.asarray(screenshot()))
            with open_pdf(path) as pdf:
                image = pdf.pages[0].Resources.XObject.Im0
                before = (bytes(image.read_bytes()), str(image.ColorSpace))
                _, changes = jpegopt.optimize_jpegs(pdf, Path(root), 0.9, 1)
                self.assertEqual(len(changes), 1)
                jpegopt.revert(changes[0])
                self.assertEqual((bytes(image.read_bytes()), str(image.ColorSpace)), before)


class XrefStreamTests(unittest.TestCase):
    def saved(self, root: str, **options) -> Path:
        source, target = Path(root) / "s.pdf", Path(root) / "t.pdf"
        make_text_pdf(source, lines=200)
        with pikepdf.open(source) as pdf:
            for number in range(300):      # many small objects: big object streams
                pdf.Root[f"/Extra{number}"] = pdf.make_indirect(pikepdf.Dictionary(
                    Index=number, Name=pikepdf.String(f"object number {number}")))
            lossless.save(pdf, target, deflated=False, **options) if options else \
                lossless.save(pdf, target, deflated=False)
        return target

    @unittest.skipUnless(lossless.ect_available(), "the ECT module is not built")
    def test_object_streams_are_packed_and_nothing_else_changes(self):
        with tempfile.TemporaryDirectory() as root:
            target = self.saved(root)
            original = target.read_bytes()
            saved = lossless.repack_xref_streams(target, 2)
            self.assertGreater(saved, 0)
            self.assertEqual(len(original) - target.stat().st_size, saved)
            reference = Path(root) / "reference.pdf"
            reference.write_bytes(original)
            self.assertTrue(lossless._same_objects(reference, target))
            with pikepdf.open(target) as pdf:
                self.assertEqual(pdf.get_warnings(), [])
                self.assertEqual(str(pdf.Root.Extra299.Name), "object number 299")
            if need("pdftoppm"):
                self.assertTrue(identical_pages(reference, target, [1], Path(root) / "x"))

    def test_files_without_an_xref_stream_and_damaged_ones_are_left_alone(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "classic.pdf"
            make_text_pdf(target, lines=50)        # pikepdf's default: xref table
            before = target.read_bytes()
            self.assertEqual(lossless.repack_xref_streams(target), 0)
            self.assertEqual(target.read_bytes(), before)
            target = self.saved(root)
            data = bytearray(target.read_bytes())
            data[data.rindex(b"startxref") + 10] = ord("9")     # a wrong offset
            target.write_bytes(bytes(data))
            self.assertEqual(lossless.repack_xref_streams(target), 0)
            self.assertEqual(target.read_bytes(), bytes(data))

    @unittest.skipUnless(lossless.ect_available(), "the ECT module is not built")
    def test_a_result_that_does_not_reopen_identically_is_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            target = self.saved(root)
            before = target.read_bytes()
            with patch.object(lossless, "_same_objects", return_value=False):
                self.assertEqual(lossless.repack_xref_streams(target), 0)
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in Path(root).iterdir()),
                             ["s.pdf", "t.pdf"])


if __name__ == "__main__":
    unittest.main()
