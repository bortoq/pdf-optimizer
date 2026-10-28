from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pikepdf
from PIL import Image

from pdfopt import imagedata, mrc
from pdfopt.pdfmodel import MRC_FORM_KEY, open_pdf
from pdfopt.render import VisualGate, identical_pages

from .helpers import make_scan_pdf, need, text_image


def gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"))


def job_for(image: Image.Image, stored=900_000, page=(360.0, 504.0)):
    data = BytesIO()
    image.save(data, "JPEG", quality=92)
    return mrc.PageJob(1, data.getvalue(), None, image.width, image.height,
                       3, stored, page)


class SegmentationTests(unittest.TestCase):
    def test_dark_text_on_white(self):
        ink = mrc.ink_mask(gray(text_image()), 200)
        self.assertTrue(0.003 < ink.mean() < 0.2)

    def test_faded_scan_is_still_segmented(self):
        faded = text_image(ink=(120, 120, 120), paper=(225, 225, 225))
        ink = mrc.ink_mask(gray(faded), 200)
        self.assertTrue(0.003 < ink.mean() < 0.2)

    def test_light_text_on_dark_paper(self):
        reversed_page = text_image(ink=(235, 235, 235), paper=(25, 25, 25))
        ink = mrc.ink_mask(gray(reversed_page), 200)
        self.assertTrue(0.003 < ink.mean() < 0.2)

    def test_blank_page_has_no_ink(self):
        blank = Image.new("L", (800, 1000), 240)
        self.assertEqual(mrc.ink_mask(np.asarray(blank), 200).sum(), 0)

    def test_specks_are_removed_at_high_resolution(self):
        page = Image.new("L", (1000, 1000), 255)
        pixels = np.asarray(page).copy()
        pixels[500, 500] = 0           # a lone dark pixel
        pixels[100:140, 100:300] = 0   # a real bar
        ink = mrc.ink_mask(pixels, 300)
        self.assertFalse(ink[500, 500])
        self.assertTrue(ink[120, 200])

    def test_ragged_edges_are_smoothed_at_high_resolution(self):
        pixels = np.full((1000, 1000), 255, dtype=np.uint8)
        pixels[100:140, 100:300] = 0
        pixels[99, 200] = 0            # a one-pixel bump on the edge
        pixels[120, 150] = 255         # a one-pixel hole inside
        ink = mrc.ink_mask(pixels, 300)
        self.assertFalse(ink[99, 200])
        self.assertTrue(ink[120, 150])


class BackgroundTests(unittest.TestCase):
    def test_ink_is_filled_with_paper_colour(self):
        paper = Image.new("RGB", (600, 600), (240, 230, 200))
        pixels = np.asarray(paper).copy()
        pixels[200:260, 100:500] = (10, 10, 10)
        bitmap = Image.fromarray(pixels)
        ink = np.zeros((600, 600), bool)
        ink[200:260, 100:500] = True
        background = mrc.background_image(
            bitmap, Image.fromarray(ink.astype(np.uint8) * 255), (300, 300))
        filled = np.asarray(background)[100:130, 50:250].astype(float)
        self.assertLess(np.abs(filled.mean(axis=(0, 1)) - (240, 230, 200)).max(), 6)

    def test_grayscale_stays_grayscale(self):
        bitmap = Image.new("L", (400, 400), 200)
        ink = Image.new("L", (400, 400), 0)
        self.assertEqual(mrc.background_image(bitmap, ink, (200, 200)).mode, "L")


class ForegroundTests(unittest.TestCase):
    def test_single_colour_text_has_no_foreground_image(self):
        bitmap = text_image()
        ink = mrc.ink_mask(gray(bitmap), 200)
        fg = mrc._foreground(np.asarray(bitmap), ink, 200)
        self.assertIsNone(fg.image)
        self.assertEqual(len(fg.inks), 1)
        self.assertLess(max(fg.inks[0][0]), 0.2)

    def test_light_text_on_dark_paper_is_one_colour(self):
        bitmap = text_image(ink=(235, 235, 235), paper=(25, 25, 25))
        ink = mrc.ink_mask(gray(bitmap), 200)
        fg = mrc._foreground(np.asarray(bitmap), ink, 200)
        self.assertIsNone(fg.image)
        self.assertGreater(min(fg.inks[0][0]), 0.85)

    def test_two_inks_get_a_stencil_each(self):
        bitmap = text_image(color=True)        # black lines and red lines
        ink = mrc.ink_mask(gray(bitmap), 200)
        fg = mrc._foreground(np.asarray(bitmap), ink, 200)
        self.assertIsNone(fg.image)
        self.assertEqual(len(fg.inks), 2)
        (dark, dark_mask), (red, red_mask) = sorted(fg.inks, key=lambda i: i[0][0])
        self.assertLess(max(dark), 0.2)
        self.assertGreater(red[0], 0.6)
        self.assertLess(red[1], 0.3)
        self.assertFalse((dark_mask & red_mask).any())
        self.assertTrue(np.array_equal(dark_mask | red_mask, ink))
        # The red lines are every third line, starting with the second one.
        rows = np.flatnonzero(red_mask.any(axis=1))
        self.assertTrue(rows.min() > np.flatnonzero(dark_mask.any(axis=1)).min())

    def test_colours_without_a_palette_get_a_colour_image(self):
        bitmap = text_image()
        pixels = np.asarray(bitmap).copy()
        # Every 40 px block of text in its own random colour: far more inks
        # than a palette holds.
        rng = np.random.default_rng(5)
        blocks = rng.integers(0, 160, (pixels.shape[0] // 40 + 1, pixels.shape[1] // 40 + 1, 3))
        colours = np.repeat(np.repeat(blocks, 40, 0), 40, 1)[:pixels.shape[0], :pixels.shape[1]]
        dark = pixels.mean(axis=2) < 128
        pixels[dark] = colours[dark]
        ink = mrc.ink_mask(gray(Image.fromarray(pixels)), 200)
        fg = mrc._foreground(pixels, ink, 200)
        self.assertIsNotNone(fg.image)
        self.assertLess(fg.image[1][0], 200)


class CompositeTests(unittest.TestCase):
    def page(self, ink_colour=(0, 0, 0)):
        bitmap = text_image(ink=ink_colour)
        ink = mrc.ink_mask(gray(bitmap), 200)
        fg = mrc._foreground(np.asarray(bitmap), ink, 200)
        return mrc._Page(bitmap, ink, fg, 100, 0.9)

    def reference(self, page, background):
        base = np.asarray(background.resize(page.view, Image.Resampling.NEAREST),
                          dtype=np.float32)
        if base.ndim == 2:
            base = base[:, :, None]
        out = base * (1 - page.coverage) + page._tint(base.shape[2]) * page.coverage
        out = out.clip(0, 255).astype(np.uint8)
        return out[:, :, 0] if out.shape[2] == 1 else out

    def test_matches_the_plain_formula_for_gray_and_colour(self):
        page = self.page()
        for mode in ("L", "RGB"):
            background = page.bitmap.convert(mode)
            for _ in range(2):   # the second call uses the cached ink layer
                self.assertTrue(np.array_equal(page.composite(background),
                                               self.reference(page, background)))

    def test_float_background_is_not_modified(self):
        page = self.page()
        values = np.full(page.bitmap.size[::-1], 200, dtype=np.float32)
        values.flags.writeable = False
        background = Image.fromarray(values, "F")
        out = page.composite(background)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, page.view[::-1])
        self.assertTrue(np.all(values == 200))


class StencilTests(unittest.TestCase):
    def test_group4_is_chosen_for_text_and_decodes_back(self):
        bitmap = text_image(600, 800)
        ink = mrc.ink_mask(gray(bitmap), 200)
        packed = np.packbits(~ink, axis=1)
        data, group4 = mrc._encode_stencil(packed, 600, 800)
        if group4:
            rows = imagedata.g4_bitmap(data, 600, 800, True)
            self.assertTrue(np.array_equal(rows >> 0, rows))
            self.assertLess(len(data), len(packed.tobytes()))

    @unittest.skipUnless(need("jbig2"), "jbig2 is not installed")
    def test_symbol_chunks_share_a_dictionary(self):
        stencils = []
        for shade in (10, 30, 50):
            bitmap = text_image(600, 800, ink=(shade,) * 3)
            ink = mrc.ink_mask(gray(bitmap), 200)
            stencils.append(imagedata.to_png(np.packbits(~ink, axis=1), 600, 800))
        with tempfile.TemporaryDirectory() as root:
            chunks = mrc.jbig2_streams(stencils, Path(root))
        self.assertEqual(len(chunks), 1)
        symbols, pages = chunks[0]
        self.assertTrue(symbols)
        self.assertEqual(len(pages), 3)

    @unittest.skipUnless(need("jbig2"), "jbig2 is not installed")
    def test_symbol_mode_passes_the_threshold(self):
        bitmap = text_image(600, 800)
        ink = mrc.ink_mask(gray(bitmap), 200)
        stencil = imagedata.to_png(np.packbits(~ink, axis=1), 600, 800)
        commands = []

        def spy(command, *args, **kwargs):
            commands.append(command)
            return mrc_run(command, *args, **kwargs)

        mrc_run = mrc.run
        with tempfile.TemporaryDirectory() as root, patch.object(mrc, "run", spy):
            mrc.jbig2_streams([stencil, stencil], Path(root), threshold=0.85)
            mrc.jbig2_streams([stencil], Path(root))
        thresholds = [command[command.index("-t") + 1] for command in commands]
        self.assertEqual(thresholds, ["0.85", "0.9"])
        # jbig2enc 0.28 (Ubuntu 24.04) rejects -t above 0.9.
        self.assertLessEqual(max(t for _, t in mrc.JBIG2_THRESHOLDS), 0.9)

    @unittest.skipUnless(need("jbig2"), "jbig2 is not installed")
    def test_lossless_threshold_uses_the_generic_coder(self):
        bitmap = text_image(600, 800)
        ink = mrc.ink_mask(gray(bitmap), 200)
        stencil = imagedata.to_png(np.packbits(~ink, axis=1), 600, 800)
        with tempfile.TemporaryDirectory() as root:
            (symbols, pages), = mrc.jbig2_streams([stencil, stencil], Path(root),
                                                  threshold=mrc.JBIG2_LOSSLESS)
        self.assertIsNone(symbols)
        self.assertEqual(len(pages), 2)

    def test_small_glyphs_get_a_safer_threshold(self):
        # 6/8 swaps appeared at 4-5 pt in a 300 dpi scan, not at 6-7 pt.
        def stem(size):
            image = text_image(1500, 1500, size=size, seed=1)
            return mrc.stem_height(mrc.ink_mask(gray(image), 300))

        self.assertEqual(mrc.symbol_threshold(stem(40)), 0.80)
        self.assertEqual(mrc.symbol_threshold(stem(14)), mrc.JBIG2_LOSSLESS)
        self.assertEqual(mrc.symbol_threshold(0.0), mrc.JBIG2_LOSSLESS)
        self.assertEqual(mrc.stem_height(np.zeros((10, 10), bool)), 0.0)

    def test_small_print_below_normal_text_gets_its_own_stencil(self):
        page = np.full((1500, 1500, 3), 255, dtype=np.uint8)
        page[:900] = np.asarray(text_image(1500, 900, size=40, seed=1))
        page[1000:] = np.asarray(text_image(1500, 500, size=14, seed=2))
        ink = mrc.ink_mask(gray(Image.fromarray(page)), 300)
        classes = mrc.size_classes(ink)
        thresholds = [threshold for threshold, _ in classes]
        self.assertEqual(thresholds[0], 0.80)
        self.assertEqual(thresholds[-1], mrc.JBIG2_LOSSLESS)
        strict = classes[-1][1]
        self.assertTrue(np.array_equal(strict[1000:], ink[1000:]))   # all small print
        self.assertGreater(classes[0][1][:900].sum(), ink[:900].sum() // 2)
        parts = np.stack([part for _, part in classes])
        self.assertTrue(np.array_equal(parts.sum(axis=0), ink))       # a partition
        self.assertEqual(mrc.size_classes(np.zeros((10, 10), bool)), [])


    @unittest.skipUnless(need("jbig2"), "jbig2 is not installed")
    def test_generic_mode_has_no_dictionary(self):
        bitmap = text_image(600, 800)
        ink = mrc.ink_mask(gray(bitmap), 200)
        stencil = imagedata.to_png(np.packbits(~ink, axis=1), 600, 800)
        with tempfile.TemporaryDirectory() as root:
            (symbols, pages), = mrc.jbig2_streams([stencil, stencil], Path(root),
                                                  symbols=False)
        self.assertIsNone(symbols)
        self.assertEqual(len(pages), 2)


class LayerTests(unittest.TestCase):
    def test_text_page_yields_small_layers(self):
        job = job_for(text_image())
        with tempfile.TemporaryDirectory() as root:
            layers = mrc.find_layers(job, 0.96, Path(root))
        self.assertIsNotNone(layers)
        self.assertLess(len(layers.background) + len(layers.inks[0].mask), 150_000)
        self.assertLessEqual(layers.bg_size[0], 1000)

    def test_dense_page_is_not_layered(self):
        rng = np.random.default_rng(1)
        noise = Image.fromarray(rng.integers(0, 255, (1000, 1400, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(mrc.find_layers(job_for(noise), 0.96, Path(root)))

    def test_no_gain_means_no_layers(self):
        job = job_for(text_image(), stored=2_000)
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(mrc.find_layers(job, 0.96, Path(root)))

    def test_blank_page_collapses_to_a_background(self):
        blank = Image.new("RGB", (1000, 1400), (245, 240, 230))
        with tempfile.TemporaryDirectory() as root:
            layers = mrc.find_layers(job_for(blank), 0.96, Path(root))
        self.assertIsNotNone(layers)
        self.assertLess(len(layers.background), 20_000)

    def test_text_page_may_use_the_text_threshold(self):
        job = job_for(text_image())
        with tempfile.TemporaryDirectory() as root:
            layers = mrc.find_layers(job, 0.96, Path(root), relaxed=0.90)
        self.assertEqual(layers.minimum, 0.90)

    def test_stencilled_photo_is_not_text_like(self):
        rng = np.random.default_rng(3)
        page = np.asarray(text_image().convert("L")).copy()
        # A textured 'photo': the stencil posterizes it, which a tile shows.
        noise = rng.normal(0, 1, (25, 40)).astype(np.float32)
        photo = np.asarray(Image.fromarray(noise, "F").resize((800, 500), Image.BICUBIC))
        page[700:1200, 100:900] = ((photo - photo.min()) / np.ptp(photo) * 230 + 15)
        for pixels, expected in ((np.asarray(text_image().convert("L")), True),
                                 (page, False)):
            bitmap = Image.fromarray(pixels)
            ink = mrc.ink_mask(pixels, 200)
            fg = mrc._foreground(np.asarray(bitmap.convert("RGB")), ink, 200)
            self.assertEqual(mrc._Page(bitmap, ink, fg, 200, 0.96).text_like(), expected)

    @unittest.skipUnless(need("opj_compress"), "opj_compress is not installed")
    def test_flat_background_prefers_jpeg2000(self):
        blank = Image.new("RGB", (2400, 3000), (245, 240, 230))
        job = job_for(blank, page=(576.0, 720.0))   # 300 dpi
        with tempfile.TemporaryDirectory() as root:
            layers = mrc.find_layers(job, 0.96, Path(root))
        self.assertIsNotNone(layers)
        self.assertEqual(layers.codec, "/JPXDecode")
        self.assertLess(len(layers.background), 2_000)


@unittest.skipUnless(need("pdftoppm"), "pdftoppm is not installed")
class BuildTests(unittest.TestCase):
    def build(self, source, root, minimum=0.96, jobs=2):
        scratch = Path(root)
        target = scratch / "mrc.pdf"
        gate = VisualGate(source, scratch, minimum, jobs)
        result = mrc.build(source, target, scratch, minimum, jobs, gate)
        return target, gate, result

    def test_layers_keep_text_and_pass_the_page_gate(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source, pages=2)
            target, gate, (passed, rejected) = self.build(source, root)
            self.assertEqual((passed, rejected), (2, 0))
            self.assertLess(target.stat().st_size, source.stat().st_size / 5)
            self.assertEqual(gate.failures(target, [1, 2]), [])
            with open_pdf(target) as pdf:
                form = pdf.pages[0].Resources.XObject.Im0
                self.assertIn(MRC_FORM_KEY, form)
                self.assertIn(b"OCR text", pdf.pages[0].Contents.read_bytes())

    def test_colour_text_scan_passes(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source, color_text=True)
            target, gate, (passed, _) = self.build(source, root)
            self.assertEqual(passed, 1)
            self.assertEqual(gate.failures(target, [1]), [])
            with open_pdf(target) as pdf:
                form = pdf.pages[0].Resources.XObject.Im0
                # Black and red text: a stencil per ink, no colour image.
                self.assertEqual(sorted(form.Resources.XObject.keys()),
                                 ["/Bg", "/Ink0", "/Ink1"])
                self.assertEqual(form.read_bytes().count(b" rg /Ink"), 2)

    def test_grayscale_and_flate_scans_are_layered(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source, gray=True, flate=True)
            target, gate, (passed, _) = self.build(source, root)
            self.assertEqual(passed, 1)
            self.assertEqual(gate.failures(target, [1]), [])

    @unittest.skipUnless(need("jbig2"), "jbig2 is not installed")
    def test_chunked_dictionaries_and_generic_mode_both_render_correctly(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source, pages=3, varied=True)
            for symbols in (True, False):
                scratch = Path(root) / str(symbols)
                scratch.mkdir()
                target = scratch / "mrc.pdf"
                gate = VisualGate(source, scratch, 0.96, 2)
                # Small test glyphs would be coded losslessly: force symbol mode.
                with patch.object(mrc, "JBIG2_CHUNK", 2), \
                        patch.object(mrc, "JBIG2_THRESHOLDS", ((0, 0.90),)):
                    mrc.build(source, target, scratch, 0.96, 2, gate, symbols=symbols)
                self.assertEqual(gate.failures(target, [1, 2, 3]), [])
                with open_pdf(target) as pdf:
                    ink = pdf.pages[0].Resources.XObject.Im0.Resources.XObject.Ink0
                    self.assertEqual(str(ink.Filter), "/JBIG2Decode")
                    parms = ink.get("/DecodeParms")
                    shared = parms is not None and "/JBIG2Globals" in parms
                    self.assertEqual(shared, symbols)

    def test_unreachable_threshold_raises(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source)
            with self.assertRaises(ValueError):
                self.build(source, root, minimum=0.9999)

    def test_no_scan_pages_raises(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source, text_page=False)   # 400 px < MIN_SIDE
            with self.assertRaises(ValueError):
                self.build(source, root)

    def test_restore_puts_the_original_image_back(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source)
            with open_pdf(source) as pdf:
                records = mrc.add_layers(pdf, Path(root), 0.96, 1)
                self.assertEqual(len(records), 1)
                self.assertEqual(pdf.pages[0].Resources.XObject.Im0.Subtype, "/Form")
                mrc.restore(records[0])
                self.assertEqual(pdf.pages[0].Resources.XObject.Im0.Subtype, "/Image")

    def test_pages_sharing_an_image_share_the_layers(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source)
            with pikepdf.open(source, allow_overwriting_input=True) as pdf:
                pdf.pages.append(pikepdf.Page(pdf.make_indirect(
                    pikepdf.Dictionary(pdf.pages[0].obj))))
                pdf.save(source)
            with open_pdf(source) as pdf:
                records = mrc.add_layers(pdf, Path(root), 0.96, 1)
                forms = [page.Resources.XObject.Im0 for page in pdf.pages]
                self.assertEqual(len(records), 2)
                self.assertEqual(forms[0].objgen, forms[1].objgen)
                # Undoing one page leaves the other layered.
                mrc.restore(records[0])
                kinds = [str(page.Resources.XObject.Im0.Subtype) for page in pdf.pages]
            self.assertEqual(kinds, ["/Image", "/Form"])

    def test_a_failing_page_does_not_cost_the_others_their_layers(self):
        real = mrc.find_layers

        def flaky(job, *args, **kwargs):
            if job.number == 2:
                raise IndexError("unexpected")
            return real(job, *args, **kwargs)

        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "scan.pdf"
            make_scan_pdf(source, pages=3, varied=True)
            with open_pdf(source) as pdf, patch.object(mrc, "find_layers", flaky):
                records = mrc.add_layers(pdf, Path(root), 0.9, 1)
        self.assertEqual(sorted(r.page for r in records), [1, 3])


@unittest.skipUnless(need("pdftoppm"), "pdftoppm is not installed")
class RestoreTests(unittest.TestCase):
    def test_pages_the_gate_rejects_return_to_the_original_image(self):
        with tempfile.TemporaryDirectory() as root:
            scratch = Path(root)
            source = scratch / "scan.pdf"
            make_scan_pdf(source, pages=3, varied=True)
            gate = VisualGate(source, scratch, 0.5, 1)

            def rejecting(target, pages, **options):
                return [2] if 2 in pages else []

            with patch.object(gate, "failures", rejecting):
                passed, rejected = mrc.build(source, scratch / "mrc.pdf", scratch, 0.5, 1, gate)
            self.assertEqual((passed, rejected), (2, 1))
            with open_pdf(source) as original, open_pdf(scratch / "mrc.pdf") as result:
                kinds = [str(p.Resources.XObject.Im0.Subtype) for p in result.pages]
                self.assertEqual(kinds, ["/Form", "/Image", "/Form"])
                self.assertEqual(
                    bytes(result.pages[1].Resources.XObject.Im0.read_raw_bytes()),
                    bytes(original.pages[1].Resources.XObject.Im0.read_raw_bytes()))
            self.assertTrue(identical_pages(source, scratch / "mrc.pdf", [2],
                                            scratch / "exact"))


@unittest.skipUnless(need("pdftoppm"), "pdftoppm is not installed")
class ColourSpaceTests(unittest.TestCase):
    def test_icc_scan_keeps_its_colour_space(self):
        with tempfile.TemporaryDirectory() as root:
            scratch = Path(root)
            source = scratch / "scan.pdf"
            make_scan_pdf(source, color_text=True, icc=True)
            gate = VisualGate(source, scratch, 0.9, 1)
            mrc.build(source, scratch / "mrc.pdf", scratch, 0.9, 1, gate)
            self.assertEqual(gate.failures(scratch / "mrc.pdf", [1]), [])
            with open_pdf(scratch / "mrc.pdf") as pdf:
                form = pdf.pages[0].Resources.XObject.Im0
                background = form.Resources.XObject.Bg
                self.assertEqual(str(background.ColorSpace[0]), "/ICCBased")
                self.assertEqual(str(form.Resources.ColorSpace.CS0[0]), "/ICCBased")
                content = form.read_bytes()
                self.assertIn(b"/CS0 cs", content)
                self.assertNotIn(b" rg ", content)

    def test_fill_colour_operators(self):
        self.assertEqual(mrc._fill_colour((0.5, 0.25, 0.0), False, False),
                         "0.50000 0.25000 0.00000 rg")
        self.assertEqual(mrc._fill_colour((0.5, 0.25, 0.0), True, False),
                         "/CS0 cs 0.50000 0.25000 0.00000 sc")
        self.assertEqual(mrc._fill_colour((0.3, 0.3, 0.3), True, True),
                         "/CS0 cs 0.30000 sc")


class PaletteTests(unittest.TestCase):
    def test_palette_colours_are_always_finite(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            colours = rng.integers(0, 255, (60, 60, 3)).astype(np.float32)
            palette, distance = mrc._palette(colours, np.ones((60, 60), bool))
            self.assertTrue(np.isfinite(palette).all())
            self.assertTrue(np.isfinite(distance).all())


if __name__ == "__main__":
    unittest.main()
