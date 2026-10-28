from itertools import pairwise
import unittest

from pdfopt import render


class RunsTests(unittest.TestCase):
    def test_pages_are_spread_over_the_cores(self):
        runs = render._runs(list(range(1, 10)), jobs=12)
        self.assertEqual(len(runs), 9)

    def test_runs_stay_consecutive_and_bounded(self):
        pages = [*range(1, 41), 50, 51]
        runs = render._runs(pages, jobs=2)
        self.assertTrue(all(len(run) <= render.CHUNK for run in runs))
        self.assertTrue(all(b - a == 1 for run in runs for a, b in pairwise(run)))
        self.assertEqual([p for run in runs for p in run], pages)

    def test_single_core_uses_full_chunks(self):
        self.assertEqual([len(r) for r in render._runs(list(range(1, 10)), jobs=1)], [8, 1])


if __name__ == "__main__":
    unittest.main()
