"""Regression tests for defects found in the audit: damaged files, PDF/A-1,
interrupted runs, output files, the no-gain cache and the quality marker."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pikepdf
from pikepdf import Array, Dictionary, Name

from pdfopt import cache, lossless, mrc
from pdfopt.cli import NO_GAIN_KNOWN, discover
from pdfopt.optimize import _keep_document_settings, optimize_pdf
from pdfopt.pdfmodel import (
    Fingerprint,
    compare,
    marker_quality,
    open_pdf,
    pdfa_part,
    read_marker,
    set_marker,
)

from .helpers import make_scan_pdf, make_text_pdf, need
from .test_cli import plain_pdf, run

XMP = (b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
       b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description '
       b'rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">%s</rdf:Description>'
       b'</rdf:RDF></x:xmpmeta><?xpacket end="w"?>')


def claim_pdfa(path: Path, part: bytes = b"<pdfaid:part>1</pdfaid:part>") -> None:
    """Rewrite ``path`` as a PDF 1.4 file whose XMP claims PDF/A."""
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        pdf.Root.Metadata = pdf.make_stream(XMP % part, Type=Name.Metadata, Subtype=Name.XML)
        pdf.save(path, force_version="1.4", fix_metadata_version=False,
                 compress_streams=False,
                 object_stream_mode=pikepdf.ObjectStreamMode.disable)


def object_streams(path: Path) -> int:
    with pikepdf.open(path) as pdf:
        return sum(1 for obj in pdf.objects
                   if isinstance(obj, pikepdf.Stream) and obj.get("/Type") == "/ObjStm")


def damage(path: Path, change) -> None:
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        change(pdf)
        pdf.save(path)


class DamagedFileTests(unittest.TestCase):
    def test_a_parent_loop_in_the_page_tree_does_not_hang(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "loop.pdf"
            make_text_pdf(book)

            def loop(pdf):
                pdf.Root.Pages.Parent = pdf.Root.Pages

            damage(book, loop)
            with open_pdf(book) as pdf:      # used to climb /Parent for ever
                self.assertEqual(len(Fingerprint.read(pdf, book).pages), 1)

    def test_a_damaged_file_is_an_error_and_the_others_are_processed(self):
        changes = {
            "rotate": lambda pdf: pdf.pages[0].obj.__setitem__("/Rotate", pikepdf.String("x")),
            "contents": lambda pdf: pdf.pages[0].obj.__setitem__("/Contents", Dictionary(A=1)),
            "number": lambda pdf: pdf.pages[0].obj.__setitem__("/Contents", Array([5])),
            "xobject": lambda pdf: pdf.pages[0].obj.__setitem__(
                "/Resources", Dictionary(XObject=Dictionary(Im0=3))),
        }
        with tempfile.TemporaryDirectory() as root:
            for name, change in changes.items():
                make_text_pdf(Path(root) / f"{name}.pdf")
                damage(Path(root) / f"{name}.pdf", change)
            good = Path(root) / "good.pdf"
            plain_pdf(good)
            before = good.stat().st_size
            for jobs in (1, 4):              # in this process and in workers
                code, out, err = run("-j", jobs, "--dry-run", root)
                self.assertEqual(code, 1)
                self.assertIn(f"errors: {len(changes)}", out)
                self.assertNotIn("Traceback", err)
                for name in changes:
                    self.assertIn(f"{name}.pdf: error:", err)
            self.assertEqual(run(good)[0], 0)
            self.assertLess(good.stat().st_size, before)


class PdfATests(unittest.TestCase):
    def test_the_claimed_part_is_read_from_both_xmp_forms(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            with open_pdf(book) as pdf:
                self.assertIsNone(pdfa_part(pdf))
            claim_pdfa(book)
            with open_pdf(book) as pdf:
                self.assertEqual(pdfa_part(pdf), 1)
            make_text_pdf(book)
            with pikepdf.open(book, allow_overwriting_input=True) as pdf:
                pdf.Root.Metadata = pdf.make_stream(
                    XMP.replace(b"%s</rdf:Description>", b"").replace(
                        b'pdfa/ns/id/">', b'pdfa/ns/id/" pdfaid:part="2"/>'))
                pdf.save(book)
            with open_pdf(book) as pdf:
                self.assertEqual(pdfa_part(pdf), 2)

    def test_pdfa1_keeps_its_version_and_gets_no_object_streams(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            claim_pdfa(book)
            status, saved = optimize_pdf(book, jobs=2)
            self.assertIn("compressed", status)
            self.assertGreater(saved, 0)
            self.assertEqual(book.read_bytes()[:8], b"%PDF-1.4")
            self.assertEqual(object_streams(book), 0)
            # Any other file still gets them.
            plain_pdf(book)
            optimize_pdf(book, jobs=2)
            self.assertGreater(object_streams(book), 0)

    @unittest.skipUnless(need("pdftoppm", "opj_compress"), "pdftoppm and OpenJPEG are needed")
    def test_pdfa1_scans_are_layered_without_jpeg_2000(self):
        def filters(path):
            with pikepdf.open(path) as pdf:
                return {str(obj.get("/Filter")) for obj in pdf.objects
                        if isinstance(obj, pikepdf.Stream)}

        with tempfile.TemporaryDirectory() as root:
            plain, archival = Path(root) / "plain.pdf", Path(root) / "archival.pdf"
            make_scan_pdf(plain)
            make_scan_pdf(archival)
            claim_pdfa(archival)
            for book in (plain, archival):
                self.assertIn("MRC", optimize_pdf(book, "lossy", min_ssim=0.9, jobs=2)[0])
            self.assertIn("/JPXDecode", filters(plain))
            self.assertNotIn("/JPXDecode", filters(archival))
            self.assertEqual(object_streams(archival), 0)


class PaletteMemoryTests(unittest.TestCase):
    def test_a_page_of_two_inks_is_split_without_comparing_every_pair(self):
        # 60 000 cells, half black and half red: comparing every far cell with
        # 2000 others took 4.6 GB and 11 s.
        rng = np.random.default_rng(1)
        cells = 60_000
        colours = np.concatenate([
            rng.normal((20, 20, 20), 6, (cells // 2, 3)),
            rng.normal((200, 30, 30), 6, (cells // 2, 3))]).reshape(cells, 1, 3)
        start = time.monotonic()
        palette, distance = mrc._palette(colours, np.ones((cells, 1), bool))
        self.assertLess(time.monotonic() - start, 5)
        self.assertEqual(len(palette), 2)
        self.assertLess(distance.max(), mrc.PALETTE_MERGE)
        self.assertEqual(sorted(round(float(c[0]), -1) for c in palette), [20, 200])


class OutputTests(unittest.TestCase):
    def test_an_existing_output_survives_a_failed_run(self):
        with tempfile.TemporaryDirectory() as root:
            book, out = Path(root) / "book.pdf", Path(root) / "out.pdf"
            book.write_bytes(b"this is not a PDF")
            out.write_bytes(b"an earlier result")
            self.assertEqual(run("-o", out, book)[0], 1)
            self.assertEqual(out.read_bytes(), b"an earlier result")
            self.assertEqual(sorted(p.name for p in Path(root).iterdir()),
                             ["book.pdf", "out.pdf"])

    def test_output_is_complete_when_it_appears(self):
        with tempfile.TemporaryDirectory() as root:
            book, out = Path(root) / "book.pdf", Path(root) / "out" / "small.pdf"
            plain_pdf(book)
            before = book.read_bytes()
            self.assertEqual(run("-o", out, book)[0], 0)
            self.assertEqual(book.read_bytes(), before)
            self.assertLess(out.stat().st_size, len(before))
            self.assertEqual([p.name for p in out.parent.iterdir()], ["small.pdf"])

    def test_work_folders_are_not_searched_for_files(self):
        with tempfile.TemporaryDirectory() as root:
            make_text_pdf(Path(root) / "book.pdf")
            for stale in (".pdfopt-abc123", "sub/.pdfopt-open-x"):
                (Path(root) / stale).mkdir(parents=True)
                make_text_pdf(Path(root) / stale / "out.pdf")
            make_text_pdf(Path(root) / "sub" / "inner.pdf")
            self.assertEqual([p.name for p in discover([Path(root)], recursive=True)],
                             ["book.pdf", "inner.pdf"])


@unittest.skipIf(sys.platform == "win32" or not hasattr(os, "geteuid") or os.geteuid() == 0,
                 "needs a folder the user cannot write to")
class ReadOnlyFolderTests(unittest.TestCase):
    def test_a_dry_run_needs_no_write_access(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "books"
            folder.mkdir()
            plain_pdf(folder / "book.pdf")
            folder.chmod(0o555)
            try:
                code, out, err = run("--dry-run", folder)
                self.assertEqual((code, err), (0, ""))
                self.assertIn("estimate", out)
                self.assertNotIn("Saved: 0 bytes", out)
                # A real run cannot replace the file and says so.
                self.assertEqual(run(folder)[0], 1)
            finally:
                folder.chmod(0o755)


class DurabilityTests(unittest.TestCase):
    def test_the_new_file_is_flushed_before_it_replaces_the_source(self):
        events = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(handle):
            events.append("fsync")
            real_fsync(handle)

        def replace(source, target):
            events.append("replace")
            real_replace(source, target)

        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            with patch("pdfopt.optimize.os.fsync", fsync), \
                    patch("pdfopt.optimize.os.replace", replace):
                self.assertIn("compressed", optimize_pdf(book, jobs=2)[0])
        self.assertEqual(events[:2], ["fsync", "replace"])


@unittest.skipIf(sys.platform == "win32", "POSIX signals")
@unittest.skipUnless(need("pdftoppm"), "pdftoppm is not installed")
class TerminationTests(unittest.TestCase):
    def terminate(self, *options, output: bool = False) -> tuple[int, str, list[str]]:
        with tempfile.TemporaryDirectory() as root:
            if output:
                options += ("-o", str(Path(root) / "out"))
            books = Path(root) / "books"
            books.mkdir()
            for index in range(3):
                make_scan_pdf(books / f"scan{index}.pdf", pages=6)
            sizes = {book.name: book.stat().st_size for book in books.iterdir()}
            process = subprocess.Popen(
                [sys.executable, "-m", "pdfopt", "-q", "90", *options, str(books)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                cwd=Path(__file__).parent.parent)
            # Stop it in the middle of a file: as soon as a work folder is there.
            deadline = time.monotonic() + 60
            while not any(p.name.startswith(".pdfopt-") for p in Path(root).rglob(".pdfopt-*")):
                if process.poll() is not None or time.monotonic() > deadline:
                    process.kill()
                    self.fail("the run made no work folder")
                time.sleep(0.02)
            process.send_signal(signal.SIGTERM)
            try:
                _, err = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                self.fail("the run did not stop after SIGTERM")
            left = sorted(str(p.relative_to(root)) for p in Path(root).rglob("*")
                          if p.name.startswith(".pdfopt-"))
            # Not one file was finished, so every source is as it was.
            self.assertEqual({book.name: book.stat().st_size for book in books.iterdir()
                              if not book.name.startswith(".")}, sizes)
            return process.returncode, err, left

    def test_a_single_process_cleans_up(self):
        code, err, left = self.terminate("-j", "1")
        self.assertEqual((code, left), (130, []))
        self.assertIn("interrupted", err)

    def test_worker_processes_are_told_and_clean_up(self):
        code, err, left = self.terminate("-j", "3", output=True)
        self.assertEqual((code, left), (130, []))
        self.assertIn("interrupted", err)


class NoGainCacheTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        environment = patch.dict(os.environ, {"XDG_CACHE_HOME": folder.name})
        environment.start()
        self.addCleanup(environment.stop)

    def book(self, root) -> Path:
        book = Path(root) / "book.pdf"
        make_text_pdf(book, lines=3)       # too small to gain anything
        return book

    def test_a_file_that_gained_nothing_is_not_analysed_again(self):
        calls = []

        def spy(path, mode, **options):
            calls.append(path)
            return optimize_pdf(path, mode, **options)

        with tempfile.TemporaryDirectory() as root, patch("pdfopt.cli.optimize_pdf", spy):
            book = self.book(root)
            before = book.read_bytes()
            self.assertIn("no change\n", run("-v", book)[1])
            self.assertIn(NO_GAIN_KNOWN, run("-v", book)[1])
            self.assertEqual(len(calls), 1)
            # Other settings, --force or other content: the work is done.
            run("-v", "-q", 90, book)
            run("-v", "--force", book)
            self.assertEqual(len(calls), 3)
            book.write_bytes(before + b"\n% changed\n")
            self.assertNotIn(NO_GAIN_KNOWN, run("-v", book)[1])
            self.assertEqual(len(calls), 4)

    def test_a_compressed_file_is_not_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            self.assertIn("possible saving", run("-v", "--dry-run", book)[1])
            self.assertEqual(cache.load(), set())

    def test_a_broken_or_unwritable_cache_is_ignored(self):
        target = cache.folder() / cache.NAME
        target.parent.mkdir(parents=True)
        target.write_text("{ not json")
        self.assertEqual(cache.load(), set())
        cache.add(["a", "b"])
        cache.add(["b", "c"])
        self.assertEqual(cache.load(), {"a", "b", "c"})
        with patch.object(cache, "LIMIT", 2):
            cache.add(["d"])
        self.assertEqual(cache.load(), {"c", "d"})
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(target)}):   # a file, not a folder
            cache.add(["e"])
            self.assertEqual(cache.load(), set())


class QualityMarkerTests(unittest.TestCase):
    def test_the_lowest_quality_is_remembered(self):
        pdf = pikepdf.new()
        set_marker(pdf, lossy=False)
        self.assertIsNone(marker_quality(pdf))
        set_marker(pdf, lossy=True, quality=96)
        self.assertEqual((read_marker(pdf), marker_quality(pdf)), ((1, True), 96))
        set_marker(pdf, lossy=True, quality=80)
        set_marker(pdf, lossy=True, quality=90)
        set_marker(pdf, lossy=False)
        self.assertEqual((read_marker(pdf), marker_quality(pdf)), ((1, True), 80))

    @unittest.skipUnless(need("pdftoppm"), "pdftoppm is not installed")
    def test_a_lower_quality_takes_a_lossy_file_again(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book)
            optimize_pdf(book, "lossy", min_ssim=0.96, jobs=2)
            with open_pdf(book) as pdf:
                self.assertEqual(marker_quality(pdf), 96)
            for same_or_higher in (0.96, 0.99):
                self.assertEqual(optimize_pdf(book, "lossy", min_ssim=same_or_higher, jobs=2),
                                 ("already optimized", 0))
            self.assertNotEqual(optimize_pdf(book, "lossy", min_ssim=0.80, jobs=2, dry_run=True),
                                ("already optimized", 0))

    def test_a_marker_without_a_quality_is_left_alone(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            with pikepdf.open(book, allow_overwriting_input=True) as pdf:
                pdf.Root.PdfOpt = Dictionary(Version=1, Lossy=True)
                pdf.save(book)
            with patch("pdfopt.optimize.which", lambda name: "/usr/bin/" + name):
                self.assertEqual(optimize_pdf(book, "lossy", min_ssim=0.5),
                                 ("already optimized", 0))


class DocumentSettingsTests(unittest.TestCase):
    def test_a_candidate_that_lost_document_features_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            full, bare = Path(root) / "full.pdf", Path(root) / "bare.pdf"
            make_text_pdf(bare)
            make_text_pdf(full)
            with pikepdf.open(full, allow_overwriting_input=True) as pdf:
                pdf.Root.Lang = "en-GB"
                pdf.Root.PageLabels = Dictionary(Nums=Array([0, Dictionary(S=Name.r)]))
                pdf.save(full)
            with open_pdf(full) as a, open_pdf(bare) as b:
                source, candidate = Fingerprint.read(a, full), Fingerprint.read(b, bare)
            compare(source, source, content=True, metadata=True)
            compare(source, candidate, content=True, metadata=True)   # gaining is fine
            with self.assertRaisesRegex(ValueError, "lost Lang, PageLabels"):
                compare(candidate, source, content=True, metadata=True)

    def test_plain_settings_and_info_are_carried_into_a_rewritten_file(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "source.pdf", Path(root) / "target.pdf"
            make_text_pdf(source)
            make_text_pdf(target)
            with pikepdf.open(source, allow_overwriting_input=True) as pdf:
                pdf.Root.Lang = "en-GB"
                pdf.Root.ViewerPreferences = Dictionary(DisplayDocTitle=True)
                pdf.docinfo["/Title"] = "My scan"
                pdf.docinfo["/Producer"] = "ScannerX"
                pdf.save(source)
            with pikepdf.open(target, allow_overwriting_input=True) as pdf:
                pdf.docinfo["/Producer"] = "Some rewriter"
                pdf.docinfo["/ModDate"] = "D:20260101"
                pdf.save(target)
            _keep_document_settings(source, target)
            with pikepdf.open(target) as pdf:
                self.assertEqual(str(pdf.Root.Lang), "en-GB")
                self.assertTrue(pdf.Root.ViewerPreferences.DisplayDocTitle)
                self.assertEqual({k: str(v) for k, v in pdf.docinfo.items()},
                                 {"/Title": "My scan", "/Producer": "ScannerX"})
            self.assertEqual([p.name for p in Path(root).iterdir() if p.name == "kept.pdf"], [])


class SaveTests(unittest.TestCase):
    def test_save_follows_the_claim_of_the_document(self):
        with tempfile.TemporaryDirectory() as root:
            book, out = Path(root) / "book.pdf", Path(root) / "out.pdf"
            make_text_pdf(book)
            claim_pdfa(book, b"<pdfaid:part>2</pdfaid:part>")
            with open_pdf(book) as pdf:
                lossless.save(pdf, out, deflated=False)
            self.assertGreater(object_streams(out), 0)     # PDF/A-2 allows them


if __name__ == "__main__":
    unittest.main()
