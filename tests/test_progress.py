from pathlib import Path
import tempfile
import unittest

from pdfopt.optimize import optimize_pdf
from pdfopt.progress import Span

from .helpers import make_scan_pdf, make_text_pdf, need


class SpanTests(unittest.TestCase):
    def test_values_never_go_down_and_are_scaled(self):
        seen = []
        span = Span(seen.append, 0.2, 0.6)
        for fraction in (0.5, 0.25, 0.5, 2.0):
            span(fraction)
        self.assertEqual(seen, [0.2 + 0.4 * 0.5, 0.6])

    def test_counter_reports_the_share_of_items_done(self):
        seen = []
        tick = Span(seen.append).counter(4)
        for _ in range(4):
            tick()
        self.assertEqual(seen, [0.25, 0.5, 0.75, 1.0])

    def test_nested_spans_map_into_the_parent(self):
        seen = []
        Span(seen.append, 0.0, 0.5).part(0.5, 1.0)(1.0)
        self.assertEqual(seen, [0.5])


class OptimizeProgressTests(unittest.TestCase):
    def test_lossless_run_reports_a_growing_fraction_up_to_one(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            seen = []
            optimize_pdf(book, "lossless", jobs=2, progress=seen.append)
            self.assertEqual(seen, sorted(seen))
            self.assertGreater(len(seen), 2)
            self.assertEqual(seen[-1], 1.0)

    @unittest.skipUnless(need("pdftoppm"), "pdftoppm is not installed")
    def test_lossy_run_reports_page_scores(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "scan.pdf"
            make_scan_pdf(book, pages=2)
            seen = []
            outcome = optimize_pdf(book, "lossy", min_ssim=0.9, jobs=2,
                                   progress=seen.append)
            self.assertEqual(seen, sorted(seen))
            self.assertEqual(len(outcome.scores), 2)
            self.assertTrue(all(0.0 <= score <= 1.0 for score in outcome.scores))

    def test_lossless_result_has_no_scores(self):
        with tempfile.TemporaryDirectory() as root:
            book = Path(root) / "book.pdf"
            make_text_pdf(book)
            self.assertEqual(optimize_pdf(book, "lossless", jobs=2).scores, ())
