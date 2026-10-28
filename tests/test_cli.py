from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pikepdf

from pdfopt.cli import discover, files_at_once, main

from .helpers import make_scan_pdf, make_text_pdf, need, sign


def run(*argv):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = main([str(a) for a in argv])
        except SystemExit as exit_:
            code = exit_.code
    return code, out.getvalue(), err.getvalue()


def plain_pdf(path: Path):
    make_text_pdf(path)
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        pdf.save(path, compress_streams=False)


class CliTests(unittest.TestCase):
    def test_lossless_dry_run_and_replacement(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            original = book.read_bytes()
            code, out, _ = run("--dry-run", book)
            self.assertEqual(code, 0)
            self.assertIn("estimate", out)
            self.assertEqual(book.read_bytes(), original)
            self.assertEqual(run(book)[0], 0)
            self.assertLess(book.stat().st_size, len(original))

    def test_quiet_run_prints_only_the_total(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            code, out, err = run(book)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            self.assertNotIn(str(book), out)
            self.assertRegex(out, r"Saved: [\d.]+ \S+ \([\d.]+%\)")

    def test_lossless_total_has_no_ssim(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            self.assertNotIn("SSIM", run(book)[1])

    @unittest.skipUnless(need("pdftoppm"), "pdftoppm is not installed")
    def test_lossy_total_shows_the_similarity(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book, pages=2)
            code, out, _ = run("-q", "90", str(book))
            self.assertEqual(code, 0)
            self.assertRegex(out, r"Saved: .* SSIM 0\.\d{4} \(lowest page 0\.\d{4}\)")

    def test_progress_bar_goes_to_a_terminal_only(self):
        class Terminal(StringIO):
            def isatty(self):
                return True

        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            out, err = StringIO(), Terminal()
            with redirect_stdout(out), redirect_stderr(err):
                main([str(book)])
            self.assertIn("1/1", err.getvalue())
            self.assertIn("%", err.getvalue())

    def test_encrypted_file_needs_decrypt_flag(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "locked.pdf"
            plain_pdf(book)
            with pikepdf.open(book, allow_overwriting_input=True) as pdf:
                pdf.save(book, compress_streams=False,
                         encryption=pikepdf.Encryption(owner="o", user=""))
            original = book.read_bytes()
            self.assertIn("--decrypt", run(book)[1])
            self.assertEqual(book.read_bytes(), original)
            self.assertEqual(run("--decrypt", book)[0], 0)
            with pikepdf.open(book) as pdf:
                self.assertFalse(pdf.is_encrypted)
                self.assertEqual(len(pdf.pages), 1)

    def test_password_protected_file_stays_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "locked.pdf"
            plain_pdf(book)
            with pikepdf.open(book, allow_overwriting_input=True) as pdf:
                pdf.save(book, encryption=pikepdf.Encryption(owner="o", user="u"))
            original = book.read_bytes()
            code, out, _ = run("--decrypt", book)
            self.assertEqual(code, 0)
            self.assertIn("password required", out)
            self.assertEqual(book.read_bytes(), original)

    def test_signed_pdf_is_skipped_with_exit_zero(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "signed.pdf"
            make_text_pdf(book)
            sign(book)
            original = book.read_bytes()
            code, out, _ = run(book)
            self.assertEqual(code, 0)
            self.assertIn("skipped", out)
            self.assertEqual(book.read_bytes(), original)

    def test_broken_pdf_gives_exit_one(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "bad.pdf"
            book.write_bytes(b"junk")
            code, _, err = run(book)
            self.assertEqual(code, 1)
            self.assertIn("error", err)

    def test_quality_out_of_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            self.assertEqual(run("-q", 0, book)[0], 2)
            self.assertEqual(run("-q", 101, book)[0], 2)
            self.assertEqual(run("-q", 90, "--mrc-quality", 0, book)[0], 2)

    def test_mrc_quality_reaches_the_optimizer(self):
        seen = []

        def fake(path, mode, **options):
            seen.append(options["mrc_ssim"])
            return "no change", 0

        with tempfile.TemporaryDirectory() as root, \
                patch("pdfopt.cli.optimize_pdf", fake):
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            run("-q", 96, "--mrc-quality", 93, book)
            run("-q", 96, book)
        self.assertEqual(seen, [0.93, None])

    def test_missing_path_returns_two(self):
        self.assertEqual(run("/definitely/not/here.pdf")[0], 2)

    def test_help_renders(self):
        code, out, _ = run("-h")
        self.assertEqual(code, 0)
        self.assertIn("--safe-text", out)
        self.assertIn("--mrc-quality", out)
        for option in ("--jobs", "--version", "--output"):
            self.assertIn(option, out)
        for removed in ("--fast", "--pdf-worker"):
            self.assertNotIn(removed, out)

    def test_removed_options_are_rejected(self):
        for option in ("--fast", "--pdf-worker"):
            self.assertEqual(run(option, "x.pdf")[0], 2, option)

    def test_missing_worker_is_only_a_warning(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            with patch("pdfopt.cli.find_worker", return_value=None):
                code, _, err = run(book)
            self.assertEqual(code, 0)
            self.assertNotIn("hpdf-worker", err)
            self.assertIn("hpdf-worker not found", run("-v", book)[2])

    def test_several_files_in_parallel(self):
        with tempfile.TemporaryDirectory() as root:
            books = [Path(root) / f"b{i}.pdf" for i in range(3)]
            for book in books:
                plain_pdf(book)
            code, out, _ = run("-v", root)
            self.assertEqual(code, 0)
            for book in books:
                self.assertIn(str(book), out)

    def test_verbose_logs_engines_to_stderr(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            _, _, err = run("-v", book)
            self.assertIn("lossless", err)

    def test_second_run_reports_already_optimized(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            run(book)
            self.assertIn("already optimized", run("-v", book)[1])
            self.assertNotIn("already optimized", run("-v", "--force", book)[1])


class OptionTests(unittest.TestCase):
    def test_version(self):
        code, out, _ = run("--version")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"^pdfopt \d+\.\d+\.\d+")

    def test_output_file_leaves_the_source_alone(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            original = book.read_bytes()
            target = Path(root) / "out" / "small.pdf"
            self.assertEqual(run("-o", target, book)[0], 0)
            self.assertEqual(book.read_bytes(), original)
            self.assertLess(target.stat().st_size, len(original))

    def test_output_directory_for_several_files(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("a.pdf", "b.pdf"):
                plain_pdf(Path(root) / name)
            target = Path(root) / "out"
            self.assertEqual(run("-o", target, Path(root) / "a.pdf", Path(root) / "b.pdf")[0], 0)
            self.assertEqual(sorted(p.name for p in target.iterdir()), ["a.pdf", "b.pdf"])

    def test_output_errors(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            plain_pdf(base / "a.pdf")
            plain_pdf(base / "b.pdf")
            (base / "file.pdf").write_bytes(b"x")
            self.assertEqual(run("-o", base / "a.pdf", base / "a.pdf")[0], 2)
            self.assertEqual(run("-o", base / "file.pdf", base / "a.pdf", base / "b.pdf")[0], 2)

    def test_output_removed_when_the_file_fails(self):
        with tempfile.TemporaryDirectory() as root:
            bad = Path(root) / "bad.pdf"
            bad.write_bytes(b"not a pdf")
            target = Path(root) / "o.pdf"
            self.assertEqual(run("-o", target, bad)[0], 1)
            self.assertFalse(target.exists())

    def test_jobs_must_be_positive(self):
        self.assertEqual(run("-j", "0", "x.pdf")[0], 2)

    def test_symlink_argument_is_explained(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            plain_pdf(base / "a.pdf")
            (base / "link.pdf").symlink_to(base / "a.pdf")
            code, _, err = run(base / "link.pdf")
            self.assertEqual(code, 2)
            self.assertIn("symbolic links", err)

    def test_already_optimized_files_are_counted_without_verbose(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            run(book)
            self.assertIn("already optimized: 1", run(book)[1])

    def test_parallel_files_are_bounded_by_memory(self):
        self.assertEqual(files_at_once(8, 10, "lossy", available_mb=1500), 2)
        self.assertEqual(files_at_once(8, 10, "lossless", available_mb=100000), 8)
        self.assertEqual(files_at_once(8, 3, "lossy", available_mb=100), 1)
        self.assertEqual(files_at_once(8, 3, "lossy", available_mb=None) >= 1, True)


class DiscoverTests(unittest.TestCase):
    def test_collects_pdf_recursively_without_symlinks(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / "sub").mkdir()
            (base / "a.pdf").write_bytes(b"x")
            (base / "b.txt").write_bytes(b"x")
            (base / "sub" / "c.PDF").write_bytes(b"x")
            (base / "link.pdf").symlink_to(base / "a.pdf")
            (base / "dirlink").symlink_to(base / "sub")
            found = discover([base], recursive=True)
            self.assertEqual({p.name for p in found}, {"a.pdf", "c.PDF"})

    def test_non_recursive_ignores_subdirectories(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / "sub").mkdir()
            (base / "a.pdf").write_bytes(b"x")
            (base / "sub" / "c.pdf").write_bytes(b"x")
            self.assertEqual([p.name for p in discover([base], False)], ["a.pdf"])

    def test_duplicates_are_collapsed(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "a.pdf"
            book.write_bytes(b"x")
            self.assertEqual(len(discover([book, book, Path(root)], False)), 1)


if __name__ == "__main__":
    unittest.main()


class InterruptTests(unittest.TestCase):
    def test_ctrl_c_stops_quietly_and_leaves_the_file_alone(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            original = book.read_bytes()
            with patch("pdfopt.cli.optimize_pdf", side_effect=KeyboardInterrupt):
                code, _, err = run(book)
            self.assertEqual(code, 130)
            self.assertIn("interrupted", err)
            self.assertNotIn("Traceback", err)
            self.assertEqual(book.read_bytes(), original)

    def test_interrupted_copy_for_output_is_removed(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            target = Path(root) / "out.pdf"
            with patch("pdfopt.cli.optimize_pdf", side_effect=KeyboardInterrupt):
                code, _, _ = run("-o", target, book)
            self.assertEqual(code, 130)
            self.assertFalse(target.exists())
