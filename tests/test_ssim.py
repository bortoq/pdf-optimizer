from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageFilter

from pdfopt.ssim import ssim

from .helpers import need


def noisy(seed=1, shape=(70, 90)):
    rng = np.random.default_rng(seed)
    base = np.add.outer(np.arange(shape[0]), np.arange(shape[1])) % 256
    base = np.stack([base, base[::-1], base], axis=2)
    return (base * 0.7 + rng.integers(0, 60, base.shape)).clip(0, 255).astype(np.uint8)


class SsimTests(unittest.TestCase):
    def test_identical_images_score_one(self):
        image = noisy()
        self.assertEqual(ssim(image, image), 1.0)

    def test_change_lowers_the_score(self):
        image = noisy()
        blurred = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(2)))
        self.assertLess(ssim(image, blurred), 0.9)

    def test_grayscale_and_color_are_supported(self):
        image = noisy()
        self.assertEqual(ssim(image[:, :, 0], image[:, :, 0]), 1.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            ssim(noisy(shape=(40, 40)), noisy(shape=(40, 48)))

    def test_tiny_images_compare_exactly(self):
        small = np.zeros((5, 5), dtype=np.uint8)
        self.assertEqual(ssim(small, small), 1.0)
        self.assertEqual(ssim(small, small + 1), 0.0)

    def test_tall_images_use_overlapping_strips(self):
        image = noisy(shape=(700, 40))
        other = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(1)))
        # A 700-row image spans three 256-row strips; the strip seams must not
        # change the result.
        whole = ssim(image, other)
        halves = [ssim(image[a:b], other[a:b]) for a, b in ((0, 352), (348, 700))]
        self.assertAlmostEqual(whole, sum(halves) / 2, delta=0.02)

    @unittest.skipUnless(need("ffmpeg"), "ffmpeg is not installed")
    def test_matches_ffmpeg(self):
        image = noisy(shape=(120, 150))
        other = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(1.5)))
        with tempfile.TemporaryDirectory() as root:
            first, second = Path(root) / "a.png", Path(root) / "b.png"
            Image.fromarray(image).save(first)
            Image.fromarray(other).save(second)
            done = subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(first), "-i",
                 str(second), "-filter_complex",
                 "[0:v]format=gbrp[a];[1:v]format=gbrp[b];[a][b]ssim",
                 "-f", "null", "-"], capture_output=True, text=True)
        expected = float(re.search(r"All:([0-9.]+)", done.stderr).group(1))
        self.assertAlmostEqual(ssim(image, other), expected, places=5)


if __name__ == "__main__":
    unittest.main()
