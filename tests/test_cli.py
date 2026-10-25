from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pikepdf

from pdfopt.cli import discover, main

from .helpers import make_text_pdf, sign


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
            self.assertIn("оценка", out)
            self.assertEqual(book.read_bytes(), original)
            self.assertEqual(run(book)[0], 0)
            self.assertLess(book.stat().st_size, len(original))

    def test_signed_pdf_is_skipped_with_exit_zero(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "signed.pdf"
            make_text_pdf(book)
            sign(book)
            original = book.read_bytes()
            code, out, _ = run(book)
            self.assertEqual(code, 0)
            self.assertIn("пропущен", out)
            self.assertEqual(book.read_bytes(), original)

    def test_broken_pdf_gives_exit_one(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "bad.pdf"
            book.write_bytes(b"junk")
            code, _, err = run(book)
            self.assertEqual(code, 1)
            self.assertIn("ошибка", err)

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
            return "без изменений", 0

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
        for removed in ("--jobs", "--fast", "--version", "--pdf-worker"):
            self.assertNotIn(removed, out)

    def test_removed_options_are_rejected(self):
        for option in ("-j", "--fast", "--version", "--pdf-worker"):
            self.assertEqual(run(option, "x.pdf")[0], 2, option)

    def test_missing_worker_is_only_a_warning(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            plain_pdf(book)
            with patch("pdfopt.cli.find_worker", return_value=None):
                code, _, err = run(book)
            self.assertEqual(code, 0)
            self.assertIn("hpdf-worker не найден", err)

    def test_several_files_in_parallel(self):
        with tempfile.TemporaryDirectory() as root:
            books = [Path(root) / f"b{i}.pdf" for i in range(3)]
            for book in books:
                plain_pdf(book)
            code, out, _ = run(root)
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
            self.assertIn("уже оптимизирован", run(book)[1])
            self.assertNotIn("уже оптимизирован", run("--force", book)[1])


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
