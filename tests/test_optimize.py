from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pikepdf

from pdfopt.optimize import MIN_SSIM, _Run, find_worker, optimize_pdf
from pdfopt.pdfmodel import Fingerprint, Skipped, open_pdf, read_marker
from pdfopt.tools import ToolError

from .helpers import (
    make_flate_image_pdf,
    make_scan_pdf,
    make_soft_scan_pdf,
    make_text_pdf,
    need,
    sign,
    stripes,
)


def worker_script(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class WorkerDiscoveryTests(unittest.TestCase):
    def test_worker_is_taken_from_path(self):
        with tempfile.TemporaryDirectory() as root:
            worker = worker_script(Path(root) / "hpdf-worker", "")
            with patch("pdfopt.optimize.which", return_value=str(worker)):
                self.assertEqual(find_worker(), worker)

    def test_absent_worker_gives_none(self):
        with patch("pdfopt.optimize.which", return_value=None):
            self.assertIsNone(find_worker())


class LosslessTests(unittest.TestCase):
    def test_dry_run_then_replacement(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            with pikepdf.open(book, allow_overwriting_input=True) as pdf:
                pdf.save(book, compress_streams=False)
            original = book.read_bytes()
            status, saved = optimize_pdf(book, "lossless", dry_run=True, jobs=2)
            self.assertGreater(saved, 0)
            self.assertIn("возможная экономия", status)
            self.assertEqual(book.read_bytes(), original)
            status, saved = optimize_pdf(book, "lossless", jobs=2)
            self.assertIn("сжато", status)
            self.assertEqual(book.stat().st_size, len(original) - saved)
            with open_pdf(book) as pdf:
                self.assertEqual(len(pdf.pages), 1)

    def test_marker_prevents_a_second_pass_until_forced(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            with pikepdf.open(book, allow_overwriting_input=True) as pdf:
                pdf.save(book, compress_streams=False)
            optimize_pdf(book, "lossless", jobs=2)
            with open_pdf(book) as pdf:
                self.assertEqual(read_marker(pdf)[1], False)
            self.assertEqual(optimize_pdf(book, "lossless", jobs=2),
                             ("уже оптимизирован", 0))
            status, _ = optimize_pdf(book, "lossless", force=True, jobs=2)
            self.assertEqual(status, "без изменений")

    def test_metadata_and_page_text_are_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_scan_pdf(book, optimize=False)
            with pikepdf.open(book, allow_overwriting_input=True) as pdf:
                pdf.docinfo["/Title"] = "Original"
                pdf.save(book, compress_streams=False)
            optimize_pdf(book, "lossless", jobs=2)
            with open_pdf(book) as pdf:
                self.assertEqual(str(pdf.docinfo["/Title"]), "Original")
                self.assertIn(b"OCR text", pdf.pages[0].Contents.read_bytes())

    def test_signed_pdf_is_skipped_and_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "signed.pdf"
            make_text_pdf(book)
            sign(book)
            original = book.read_bytes()
            with self.assertRaises(Skipped):
                optimize_pdf(book, "lossless", jobs=2)
            self.assertEqual(book.read_bytes(), original)

    def test_unreadable_pdf_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "bad.pdf"
            book.write_bytes(b"%PDF-1.4 nothing here")
            with self.assertRaises(ValueError):
                optimize_pdf(book, "lossless", jobs=2)

    def test_unknown_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            with self.assertRaises(ValueError):
                optimize_pdf(book, "turbo")

    def test_file_changed_during_processing_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            with pikepdf.open(book, allow_overwriting_input=True) as pdf:
                pdf.save(book, compress_streams=False)
            touched = []

            def log(message):
                if not touched:
                    touched.append(True)
                    with book.open("ab") as stream:
                        stream.write(b"\n%changed\n")

            with self.assertRaises(ValueError) as caught:
                optimize_pdf(book, "lossless", jobs=2, log=log)
            self.assertIn("изменился", str(caught.exception))
            self.assertTrue(book.read_bytes().endswith(b"%changed\n"))

    def test_no_temporary_files_are_left_behind(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            optimize_pdf(book, "lossless", jobs=2)
            self.assertEqual([p.name for p in Path(root).iterdir()], ["book.pdf"])


@unittest.skipUnless(need("pdftoppm"), "pdftoppm не установлен")
class FallbackTests(unittest.TestCase):
    def test_without_a_renderer_image_rewrites_are_dropped_not_the_file(self):
        gray = stripes(gray=True)
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "rgb.pdf"
            make_flate_image_pdf(book, np.stack([gray] * 3, axis=2))
            before = book.stat().st_size
            with patch("pdfopt.render.require", side_effect=ToolError("pdftoppm не найден")):
                status, saved = optimize_pdf(book, "lossless", jobs=2)
            self.assertIn("lossless-safe", status)
            with open_pdf(book) as pdf:
                self.assertEqual(str(pdf.pages[0].Resources.XObject.Im0.ColorSpace),
                                 "/DeviceRGB")
            self.assertEqual(book.stat().st_size, before - saved)


class WorkerTests(unittest.TestCase):
    def test_failing_worker_does_not_break_the_file(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            worker = worker_script(Path(root) / "w", "import sys; sys.exit(3)")
            with patch("pdfopt.optimize.find_worker", return_value=worker):
                status, _ = optimize_pdf(book, "lossless", jobs=1)
            self.assertIn("сжато", status)

    @unittest.skipUnless(need("pdftoppm"), "pdftoppm не установлен")
    def test_lossless_worker_output_is_verified(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book, lines=20)
            # A "worker" that silently destroys the page.
            worker = worker_script(Path(root) / "w", (
                "import sys, pikepdf\n"
                "pdf = pikepdf.new()\n"
                "pdf.add_blank_page(page_size=(612, 792))\n"
                "pdf.save(sys.argv[2])\n"))
            before = book.read_bytes()
            with patch("pdfopt.optimize.find_worker", return_value=worker):
                optimize_pdf(book, "lossless", jobs=1, dry_run=True)
            self.assertEqual(book.read_bytes(), before)
            with open_pdf(book) as pdf:
                self.assertIsNone(read_marker(pdf))


@unittest.skipUnless(need("pdftoppm"), "pdftoppm не установлен")
class LossyTests(unittest.TestCase):
    def test_scan_is_layered_and_text_survives(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book, pages=2)
            before = book.stat().st_size
            status, saved = optimize_pdf(book, "lossy", min_ssim=0.96, jobs=2)
            self.assertIn("MRC", status)
            self.assertGreater(saved, before * 0.7)
            with open_pdf(book) as pdf:
                self.assertEqual(len(pdf.pages), 2)
                self.assertEqual(read_marker(pdf)[1], True)
                self.assertIn(b"OCR text", pdf.pages[1].Contents.read_bytes())

    def test_lossy_runs_once(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book)
            optimize_pdf(book, "lossy", min_ssim=0.96, jobs=2)
            self.assertEqual(optimize_pdf(book, "lossy", min_ssim=0.96, jobs=2),
                             ("уже оптимизирован", 0))
            # A lossless pass over a lossy file is also pointless.
            self.assertEqual(optimize_pdf(book, "lossless", jobs=2),
                             ("уже оптимизирован", 0))

    def test_lossless_marked_file_still_gets_the_lossy_pass(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book, optimize=False)
            optimize_pdf(book, "lossless", jobs=2)
            with open_pdf(book) as pdf:
                self.assertEqual(read_marker(pdf)[1], False)
            status, _ = optimize_pdf(book, "lossy", min_ssim=0.96, jobs=2)
            self.assertIn("сжато", status)

    def test_unreachable_quality_keeps_the_lossless_result(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book, optimize=False)
            status, _ = optimize_pdf(book, "lossy", min_ssim=1.0, jobs=2)
            self.assertIn("lossless", status)

    def test_soft_scan_is_layered_under_the_text_threshold(self):
        with tempfile.TemporaryDirectory() as root:
            sizes = {}
            for mrc_ssim in (0.96, None):     # same as -q, then the default
                book = Path(root) / f"scan-{mrc_ssim}.pdf"
                make_soft_scan_pdf(book)
                status, _ = optimize_pdf(book, "lossy", min_ssim=0.96,
                                         mrc_ssim=mrc_ssim, jobs=2)
                sizes[mrc_ssim] = (status, book.stat().st_size)
            self.assertNotIn("MRC", sizes[0.96][0])
            self.assertEqual(sizes[None][0], "сжато (MRC)")
            self.assertLess(sizes[None][1] * 5, sizes[0.96][1])

    def test_default_threshold_constant(self):
        self.assertAlmostEqual(MIN_SSIM, 0.96)

    def test_signed_scan_is_never_touched(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book)
            sign(book)
            before = book.read_bytes()
            with self.assertRaises(Skipped):
                optimize_pdf(book, "lossy", jobs=2)
            self.assertEqual(book.read_bytes(), before)


class RobustnessTests(unittest.TestCase):
    def test_a_crashing_mrc_still_leaves_the_jpeg_stage(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book, pages=2, varied=True, optimize=False)
            with patch("pdfopt.mrc.build", side_effect=RuntimeError("bug")):
                status, saved = optimize_pdf(book, "lossy", min_ssim=0.9, jobs=1)
            self.assertIn("JPEG", status)
            self.assertGreater(saved, 0)

    def test_a_crashing_jpeg_stage_keeps_the_verified_mrc(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book, pages=2, varied=True)
            with patch("pdfopt.optimize._Run.jpeg_stage", side_effect=RuntimeError("bug")):
                status, _ = optimize_pdf(book, "lossy", min_ssim=0.9, jobs=1)
            self.assertEqual(status, "сжато (MRC)")


@unittest.skipUnless(need("pdftoppm"), "pdftoppm не установлен")
class JpegRollbackTests(unittest.TestCase):
    def test_images_of_rejected_pages_are_put_back(self):
        with tempfile.TemporaryDirectory() as root:
            scratch = Path(root)
            source = scratch / "scan.pdf"
            make_scan_pdf(source, pages=3, varied=True)
            with open_pdf(source) as pdf:
                originals = [bytes(p.Resources.XObject.Im0.read_raw_bytes())
                             for p in pdf.pages]
                run = _Run(source, "lossy", 0.5, 1, scratch,
                           Fingerprint.read(pdf, source), lambda message: None)
            gate = run.visual_gate()
            with patch.object(gate, "failures", lambda target, pages, **options: [1]), \
                    open_pdf(source) as pdf:
                self.assertTrue(run.jpeg_stage(pdf, scratch / "out.pdf", scratch))
            with open_pdf(scratch / "out.pdf") as pdf:
                now = [bytes(p.Resources.XObject.Im0.read_raw_bytes()) for p in pdf.pages]
            self.assertEqual(now[0], originals[0])
            self.assertNotEqual(now[1], originals[1])
            # Nothing passes: nothing is saved and every image is original.
            with patch.object(gate, "failures", lambda target, pages, **options: list(pages)), \
                    open_pdf(source) as pdf:
                self.assertFalse(run.jpeg_stage(pdf, scratch / "none.pdf", scratch))
                self.assertEqual([bytes(p.Resources.XObject.Im0.read_raw_bytes())
                                  for p in pdf.pages], originals)


@unittest.skipUnless(need("gs", "pdftoppm"), "Ghostscript не установлен")
class GhostscriptTests(unittest.TestCase):
    def runs(self, ocr: bool) -> list:
        import pdfopt.optimize as optimize
        calls = []
        real = optimize._make_ghostscript

        def spy(source, target, dpi, qfactor):
            calls.append(dpi)
            real(source, target, dpi, qfactor)

        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book, ocr=ocr, varied=True)
            with patch.object(optimize, "_make_ghostscript", spy):
                optimize_pdf(book, "lossy", min_ssim=0.85, jobs=4, dry_run=True)
        return calls

    def test_offered_only_to_documents_without_text(self):
        self.assertEqual(sorted(self.runs(ocr=False)), [200, 240, 300])
        self.assertEqual(self.runs(ocr=True), [])


if __name__ == "__main__":
    unittest.main()
