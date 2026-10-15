from pathlib import Path
from io import BytesIO
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject,
                           EncodedStreamObject, NameObject, NumberObject)
from PIL import Image, ImageDraw, ImageFont

from pdfopt.cli import main, discover
from pdfopt.optimize import (MIN_SSIM, _find_worker, _jpeg_eligible,
                             _jpegtran_encode, _make_jpegtran_candidate,
                             _make_mrc_candidate, _make_pypdf_candidate,
                             _page_ssim, _pdf_snapshot, _render_page,
                             _render_pages, _same_pixels, optimize_pdf)


def make_pdf(path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    stream = DecodedStreamObject()
    stream.set_data(b"q\n" + b"0 0 m 100 100 l S\n" * 5000 + b"Q\n")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def make_scanned_pdf(path: Path, *, text_page: bool = False,
                     optimize: bool = True) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=400)
    image = EncodedStreamObject()
    if text_page:
        pixels = Image.new("RGB", (1200, 1600), "white")
        draw = ImageDraw.Draw(pixels)
        font = ImageFont.load_default(size=24)
        for y in range(80, 1500, 42):
            draw.text((90, y), "Pdfopt scanned text 12345",
                      fill=(10, 10, 10), font=font)
    else:
        pixels = Image.new("RGB", (400, 500))
        pixels.putdata([((x // 8) % 256, (y // 8) % 256, ((x + y) // 8) % 256)
                        for y in range(500) for x in range(400)])
    encoded = BytesIO()
    pixels.save(encoded, "JPEG", quality=95, optimize=optimize)
    image._data = encoded.getvalue()
    image.update({
        NameObject("/Type"): NameObject("/XObject"),
        NameObject("/Subtype"): NameObject("/Image"),
        NameObject("/Filter"): NameObject("/DCTDecode"),
        NameObject("/Width"): NumberObject(pixels.width),
        NameObject("/Height"): NumberObject(pixels.height),
        NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
        NameObject("/BitsPerComponent"): NumberObject(8),
    })
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/XObject"): DictionaryObject({
            NameObject("/Im0"): writer._add_object(image),
        }),
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): writer._add_object(font),
        }),
    })
    contents = DecodedStreamObject()
    contents.set_data(b"q 300 0 0 400 0 0 cm /Im0 Do Q "
                      b"BT /F1 12 Tf 50 50 Td (OCR text) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(contents)
    with path.open("wb") as output:
        writer.write(output)


class WorkerDiscoveryTests(unittest.TestCase):
    def test_explicit_worker_missing_raises(self):
        with tempfile.TemporaryDirectory() as root:
            missing = Path(root) / "hpdf-worker"
            with self.assertRaises(ValueError):
                _find_worker(missing)

    def test_explicit_worker_returned(self):
        with tempfile.TemporaryDirectory() as root:
            worker = Path(root) / "hpdf-worker"
            worker.write_bytes(b"#!/bin/sh\n")
            worker.chmod(0o755)
            found, explicit = _find_worker(worker)
            self.assertEqual(found, worker)
            self.assertTrue(explicit)

    def test_env_worker_reported_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            worker = Path(root) / "hpdf-worker"
            worker.write_bytes(b"#!/bin/sh\n")
            worker.chmod(0o755)
            with patch.dict(os.environ, {"PDFOPT_HYPER_WORKER": str(worker)}, clear=False):
                with patch("pdfopt.optimize.shutil.which", return_value=None):
                    found, explicit = _find_worker(None)
            self.assertEqual(found, worker)
            self.assertTrue(explicit)

    def test_env_worker_unavailable_raises(self):
        with patch.dict(os.environ, {"PDFOPT_HYPER_WORKER": "/nope/hpdf-worker"}, clear=False):
            with self.assertRaises(ValueError):
                _find_worker(None)

    def test_absent_worker_is_optional(self):
        env = {key: "" for key in
               ("PDFOPT_HYPER_WORKER", "BIBLIOPTER_HYPER_WORKER", "HYPER_DRV")}
        with patch.dict(os.environ, env, clear=False):
            with patch("pdfopt.optimize.shutil.which", return_value=None):
                found, explicit = _find_worker(None)
        self.assertIsNone(found)
        self.assertFalse(explicit)


class PypdfCandidateTests(unittest.TestCase):
    def test_candidate_is_smaller_and_readable(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "book.pdf"
            target = Path(root) / "out.pdf"
            make_pdf(source)
            _make_pypdf_candidate(source, target)
            self.assertLess(target.stat().st_size, source.stat().st_size)
            with target.open("rb") as stream:
                self.assertEqual(len(PdfReader(stream, strict=True).pages), 1)


def jpeg_object(**overrides):
    image = DictionaryObject({
        NameObject("/Filter"): NameObject("/DCTDecode"),
        NameObject("/BitsPerComponent"): NumberObject(8),
    })
    for key, value in overrides.items():
        image[NameObject("/" + key)] = value
    return image


class JpegEligibilityTests(unittest.TestCase):
    def test_plain_dct_is_eligible(self):
        self.assertTrue(_jpeg_eligible(jpeg_object()))

    def test_flate_chain_is_eligible(self):
        self.assertTrue(_jpeg_eligible(jpeg_object(
            Filter=ArrayObject([NameObject("/FlateDecode"), NameObject("/DCTDecode")]))))

    def test_flate_only_is_not_eligible(self):
        self.assertFalse(_jpeg_eligible(jpeg_object(Filter=NameObject("/FlateDecode"))))

    def test_non_8_bit_is_not_eligible(self):
        self.assertFalse(_jpeg_eligible(jpeg_object(BitsPerComponent=NumberObject(16))))

    def test_masks_and_decode_stay_eligible(self):
        self.assertTrue(_jpeg_eligible(jpeg_object(SMask=NameObject("/X"))))
        self.assertTrue(_jpeg_eligible(jpeg_object(Mask=NameObject("/X"))))
        self.assertTrue(_jpeg_eligible(jpeg_object(Decode=NumberObject(1))))

    def test_decode_parms_is_not_eligible(self):
        self.assertFalse(_jpeg_eligible(jpeg_object(DecodeParms=DictionaryObject())))


class SamePixelsTests(unittest.TestCase):
    @staticmethod
    def _jpeg(color, size=(32, 32)):
        buffer = BytesIO()
        Image.new("RGB", size, color).save(buffer, "JPEG", quality=90)
        return buffer.getvalue()

    def test_identical_bytes_match(self):
        data = self._jpeg((10, 20, 30))
        self.assertTrue(_same_pixels(data, data))

    def test_different_pixels_do_not_match(self):
        self.assertFalse(_same_pixels(self._jpeg((10, 20, 30)),
                                      self._jpeg((200, 10, 10))))

    def test_invalid_data_is_rejected(self):
        self.assertFalse(_same_pixels(b"nope", b"nope"))


@unittest.skipUnless(shutil.which("jpegtran"), "jpegtran не установлен")
class JpegtranTests(unittest.TestCase):
    @staticmethod
    def _jpeg(size=(256, 256)):
        image = Image.new("RGB", size)
        image.putdata([((x * 3) % 256, (y * 5) % 256, (x + y) % 256)
                       for y in range(size[1]) for x in range(size[0])])
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=92)
        return buffer.getvalue()

    def test_encode_keeps_pixels_when_smaller(self):
        data = self._jpeg()
        with tempfile.TemporaryDirectory() as root:
            encoded = _jpegtran_encode(data, Path(root), shutil.which("jpegtran"))
        if encoded is not None:
            self.assertLess(len(encoded), len(data))
            self.assertTrue(_same_pixels(data, encoded))

    def test_candidate_preserves_page_and_pixels(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source = directory / "scan.pdf"
            target = directory / "out.pdf"
            make_scanned_pdf(source, text_page=True, optimize=False)
            with source.open("rb") as stream:
                original = PdfReader(stream).pages[0]["/Resources"]["/XObject"]["/Im0"]._data
            _make_jpegtran_candidate(source, target, directory)
            with target.open("rb") as stream:
                page = PdfReader(stream, strict=True).pages[0]
                self.assertEqual(page.extract_text().strip(), "OCR text")
                result = page["/Resources"]["/XObject"]["/Im0"]._data
            self.assertTrue(_same_pixels(original, result))


class SnapshotTests(unittest.TestCase):
    def test_snapshot_reports_single_page_without_interactive(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_pdf(book)
            pages, metadata, interactive = _pdf_snapshot(book)
            self.assertEqual(len(pages), 1)
            self.assertFalse(interactive)


class CliTests(unittest.TestCase):
    def test_lossless_pdf_dry_run_and_replacement(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_pdf(book)
            original = book.read_bytes()
            self.assertEqual(main(["--dry-run", str(book)]), 0)
            self.assertEqual(book.read_bytes(), original)
            self.assertEqual(main([str(book)]), 0)
            self.assertLess(book.stat().st_size, len(original))
            with book.open("rb") as stream:
                reader = PdfReader(stream, strict=True)
                self.assertEqual(len(reader.pages), 1)
                self.assertEqual(float(reader.pages[0].mediabox.width), 612)

    def test_signed_pdf_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "signed.pdf"
            make_pdf(book)
            with book.open("ab") as output:
                output.write(b"\n% /ByteRange [0 1 2 3]\n")
            original = book.read_bytes()
            self.assertEqual(main([str(book)]), 1)
            self.assertEqual(book.read_bytes(), original)

    def test_quality_with_lossless_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_pdf(book)
            self.assertEqual(main(["-q", "96", "--lossless", str(book)]), 2)

    def test_quality_out_of_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_pdf(book)
            self.assertEqual(main(["-q", "0", str(book)]), 2)

    def test_missing_path_returns_two(self):
        self.assertEqual(main(["/definitely/missing/book.pdf"]), 2)


class DiscoverTests(unittest.TestCase):
    def test_collects_pdf_recursively_without_symlinks(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "a.pdf").write_bytes(b"%PDF")
            (directory / "b.txt").write_text("x")
            nested = directory / "sub"
            nested.mkdir()
            (nested / "c.PDF").write_bytes(b"%PDF")
            try:
                (directory / "link.pdf").symlink_to(directory / "a.pdf")
            except OSError:
                pass
            found = discover([directory], True)
            self.assertEqual(sorted(p.name for p in found), ["a.pdf", "c.PDF"])

    def test_non_recursive_ignores_subdirectories(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "a.pdf").write_bytes(b"%PDF")
            nested = directory / "sub"
            nested.mkdir()
            (nested / "c.pdf").write_bytes(b"%PDF")
            found = discover([directory], False)
            self.assertEqual([p.name for p in found], ["a.pdf"])


class SsimTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg не установлен")
    def test_identical_images_score_one_and_change_scores_less(self):
        with tempfile.TemporaryDirectory() as root:
            before = Path(root) / "before.png"
            same = Path(root) / "same.png"
            after = Path(root) / "after.png"
            image = Image.new("RGB", (128, 128), (240, 230, 220))
            for x in range(20, 110):
                for y in range(20, 110):
                    image.putpixel((x, y), (x * 2, y * 2, (x + y) % 256))
            image.save(before)
            image.save(same)
            changed = image.copy()
            for x in range(20, 110):
                for y in range(20, 110):
                    changed.putpixel((x, y), (0, 0, 0))
            changed.save(after)
            self.assertEqual(_page_ssim(before, same), 1.0)
            self.assertLess(_page_ssim(before, after), 1.0)

    @unittest.skipUnless(shutil.which("pdftoppm"), "pdftoppm не установлен")
    def test_render_page_rejects_unknown_suffix(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            book = directory / "book.pdf"
            make_pdf(book)
            with self.assertRaises(ValueError):
                _render_page(book, 1, directory / "page.bmp")


class LossyToolTests(unittest.TestCase):
    @unittest.skipUnless(all(shutil.which(command) for command in
                             ("gs", "pdftoppm", "ffmpeg")),
                         "PDF lossy tools are not installed")
    def test_lossy_pdf_recompresses_image_with_page_ssim_gate(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            book = directory / "scan.pdf"
            make_scanned_pdf(book)
            original_size = book.stat().st_size
            with book.open("rb") as stream:
                original_image = PdfReader(stream).pages[0]["/Resources"]["/XObject"]["/Im0"]._data
            original_page = _render_pages(book, directory / "before", 1)[0]

            self.assertEqual(main(["-q", "96", str(book)]), 0)
            self.assertLess(book.stat().st_size, original_size)
            result_page = _render_pages(book, directory / "after", 1)[0]
            self.assertGreaterEqual(_page_ssim(original_page, result_page), MIN_SSIM)
            with book.open("rb") as stream:
                page = PdfReader(stream).pages[0]
                self.assertEqual(page.extract_text().strip(), "OCR text")
                objects = page["/Resources"]["/XObject"]
                image = next(iter(objects.values()))
                self.assertEqual(image.get_object()["/Filter"], "/DCTDecode")
                self.assertLess(len(image.get_object()._data), len(original_image))

            make_scanned_pdf(book)
            self.assertEqual(main(["-q", "100", str(book)]), 0)
            with book.open("rb") as stream:
                objects = PdfReader(stream).pages[0]["/Resources"]["/XObject"]
                image = next(iter(objects.values()))
                self.assertEqual(image.get_object()._data, original_image)


class MrcToolTests(unittest.TestCase):
    @unittest.skipUnless(all(shutil.which(command) for command in
                             ("pdftoppm", "ffmpeg")),
                         "PDF MRC verification tools are not installed")
    def test_mrc_layers_preserve_ocr_and_pass_page_ssim(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            original = directory / "scan.pdf"
            candidate = directory / "layered.pdf"
            make_scanned_pdf(original, text_page=True)
            before = _render_pages(original, directory / "before", 1)[0]
            _make_mrc_candidate(original, candidate, directory)
            self.assertLess(candidate.stat().st_size, original.stat().st_size)
            after = _render_pages(candidate, directory / "after", 1)[0]
            self.assertGreaterEqual(_page_ssim(before, after), MIN_SSIM)
            with candidate.open("rb") as stream:
                page = PdfReader(stream).pages[0]
                self.assertEqual(page.extract_text().strip(), "OCR text")
                form = next(iter(page["/Resources"]["/XObject"].values())).get_object()
                self.assertEqual(form["/Subtype"], "/Form")
                layers = form["/Resources"]["/XObject"]
                self.assertEqual(layers["/Bg"]["/Filter"], "/DCTDecode")
                self.assertTrue(layers["/Ink"]["/ImageMask"])

    @unittest.skipUnless(all(shutil.which(command) for command in
                             ("pdftoppm", "ffmpeg")),
                         "PDF MRC verification tools are not installed")
    def test_mrc_reverts_only_page_below_ssim_threshold(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            first, second = directory / "first.pdf", directory / "second.pdf"
            source, target = directory / "both.pdf", directory / "result.pdf"
            make_scanned_pdf(first, text_page=True)
            make_scanned_pdf(second, text_page=True)
            writer = PdfWriter()
            writer.append(str(first))
            writer.append(str(second))
            with source.open("wb") as output:
                writer.write(output)
            with patch("pdfopt.optimize._page_ssim", side_effect=(.99, .99, .99, .10)):
                _make_mrc_candidate(source, target, directory)
            with target.open("rb") as stream:
                pages = PdfReader(stream).pages
                types = [next(iter(page["/Resources"]["/XObject"].values()))
                         .get_object()["/Subtype"] for page in pages]
                self.assertEqual(types, ["/Form", "/Image"])


class OptimizeApiTests(unittest.TestCase):
    def test_optimize_returns_status_and_saving(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_pdf(book)
            status, saving = optimize_pdf(book, "lossless", True)
            self.assertIn("возможная экономия", status)
            self.assertGreater(saving, 0)


if __name__ == "__main__":
    unittest.main()
