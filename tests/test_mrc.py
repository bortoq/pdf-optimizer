from pathlib import Path
from io import BytesIO
import shutil
import tempfile
import unittest

from PIL import Image
from pypdf.generic import ArrayObject, NameObject, NumberObject

from pdfopt.mrc import (_eligible, _encode_stencil, _globals_stream,
                        _image_stream, _jbig2_streams, _stencil_png,
                        encode_jpeg_trials, encode_jpx_trials)


def base_image(**overrides):
    image = _image_stream(b"payload", 800, 900, stencil=False)
    for key, value in overrides.items():
        image[NameObject("/" + key)] = value
    return image


class ImageStreamTests(unittest.TestCase):
    def test_background_stream_is_dct_rgb(self):
        image = _image_stream(b"jpeg", 640, 480, stencil=False)
        self.assertEqual(image["/Subtype"], "/Image")
        self.assertEqual(image["/Filter"], "/DCTDecode")
        self.assertEqual(image["/ColorSpace"], "/DeviceRGB")
        self.assertEqual(image["/BitsPerComponent"], 8)
        self.assertNotIn("/ImageMask", image)

    def test_stencil_stream_is_one_bit_flate_mask(self):
        image = _image_stream(b"mask", 640, 480, stencil=True)
        self.assertEqual(image["/Filter"], "/FlateDecode")
        self.assertEqual(image["/BitsPerComponent"], 1)
        self.assertTrue(image["/ImageMask"])
        self.assertNotIn("/ColorSpace", image)

    def test_group4_stencil_carries_decode_parameters(self):
        image = _image_stream(b"mask", 640, 480, stencil=True, group4=True)
        self.assertEqual(image["/Filter"], "/CCITTFaxDecode")
        params = image["/DecodeParms"]
        self.assertEqual(params["/K"], -1)
        self.assertEqual(params["/Columns"], 640)
        self.assertEqual(params["/Rows"], 480)
        self.assertTrue(params["/BlackIs1"])

    def test_jbig2_stencil_carries_global_reference(self):
        reference = NumberObject(7)
        image = _image_stream(b"mask", 640, 480, stencil=True,
                              jbig2_globals=reference)
        self.assertEqual(image["/Filter"], "/JBIG2Decode")
        self.assertTrue(image["/ImageMask"])
        self.assertIs(image["/DecodeParms"]["/JBIG2Globals"], reference)

    def test_jpx_background_stream(self):
        image = _image_stream(b"codestream", 640, 480, stencil=False, jpx=True)
        self.assertEqual(image["/Filter"], "/JPXDecode")
        self.assertEqual(image["/ColorSpace"], "/DeviceRGB")
        self.assertNotIn("/ImageMask", image)


class GlobalsStreamTests(unittest.TestCase):
    def test_keeps_raw_bytes_without_a_filter(self):
        stream = _globals_stream(b"globals")
        self.assertEqual(stream._data, b"globals")
        self.assertNotIn("/Filter", stream)


class EligibilityTests(unittest.TestCase):
    def test_large_dct_rgb_image_is_eligible(self):
        self.assertTrue(_eligible(base_image()))

    def test_flate_then_dct_chain_is_eligible(self):
        image = base_image(Filter=ArrayObject(
            [NameObject("/FlateDecode"), NameObject("/DCTDecode")]))
        self.assertTrue(_eligible(image))

    def test_plain_flate_is_not_eligible(self):
        self.assertFalse(_eligible(base_image(Filter=NameObject("/FlateDecode"))))

    def test_small_images_are_not_eligible(self):
        self.assertFalse(_eligible(base_image(Width=NumberObject(499))))
        self.assertFalse(_eligible(base_image(Height=NumberObject(499))))
        self.assertTrue(_eligible(base_image(Width=NumberObject(500),
                                             Height=NumberObject(500))))

    def test_non_rgb_or_non_8bit_is_not_eligible(self):
        self.assertFalse(_eligible(base_image(ColorSpace=NameObject("/DeviceGray"))))
        self.assertFalse(_eligible(base_image(BitsPerComponent=NumberObject(16))))

    def test_masked_images_are_not_eligible(self):
        for key in ("Mask", "SMask", "Decode", "DecodeParms"):
            self.assertFalse(_eligible(base_image(**{key: NameObject("/X")})),
                             f"{key} должен исключать изображение")


class JpegTrialTests(unittest.TestCase):
    @staticmethod
    def _bitmap():
        image = Image.new("RGB", (64, 64))
        image.putdata([((x * 4) % 256, (y * 4) % 256, (x + y) % 256)
                       for y in range(64) for x in range(64)])
        return image

    def test_returns_sorted_decodable_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            trials = encode_jpeg_trials(self._bitmap(), 75, Path(root))
        self.assertGreaterEqual(len(trials), 1)
        self.assertEqual(trials, sorted(trials, key=len))
        for data in trials:
            with Image.open(BytesIO(data)) as image:
                self.assertEqual(image.size, (64, 64))

    @unittest.skipUnless(shutil.which("cjpeg"), "cjpeg не установлен")
    def test_cjpeg_adds_a_candidate_not_larger_than_pillow(self):
        with tempfile.TemporaryDirectory() as root:
            trials = encode_jpeg_trials(self._bitmap(), 75, Path(root))
        self.assertGreaterEqual(len(trials), 2)
        self.assertLessEqual(len(trials[0]), len(trials[-1]))


class JpxTrialTests(unittest.TestCase):
    @staticmethod
    def _bitmap():
        image = Image.new("RGB", (128, 96))
        image.putdata([((x * 2) % 256, (y * 3) % 256, (x + y) % 256)
                       for y in range(96) for x in range(128)])
        return image

    @unittest.skipUnless(shutil.which("opj_compress"), "opj_compress не установлен")
    def test_returns_sorted_decodable_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            trials = encode_jpx_trials(self._bitmap(), 75, Path(root))
        self.assertGreaterEqual(len(trials), 1)
        self.assertEqual(trials, sorted(trials, key=len))
        for data in trials:
            with Image.open(BytesIO(data)) as image:
                self.assertEqual(image.size, (128, 96))


class Jbig2Tests(unittest.TestCase):
    @staticmethod
    def _stencil(offset=0):
        image = Image.new("1", (128, 128), 1)
        for y in range(16 + offset, 64, 8):
            for x in range(16, 112):
                image.putpixel((x, y), 0)
        return image

    def test_stencil_png_roundtrips(self):
        stencil = self._stencil()
        with Image.open(BytesIO(_stencil_png(stencil))) as decoded:
            self.assertEqual(decoded.size, stencil.size)

    @unittest.skipUnless(shutil.which("jbig2"), "jbig2 не установлен")
    def test_streams_cover_every_page(self):
        stencils = [_stencil_png(self._stencil(0)), _stencil_png(self._stencil(3))]
        with tempfile.TemporaryDirectory() as root:
            result = _jbig2_streams(stencils, Path(root))
        self.assertIsNotNone(result)
        globals_bytes, pages = result
        self.assertGreater(len(globals_bytes), 0)
        self.assertEqual(len(pages), 2)
        for page in pages:
            self.assertGreater(len(page), 0)

    def test_missing_inputs_returns_none(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(_jbig2_streams([], Path(root)))


class StencilEncodingTests(unittest.TestCase):
    def test_returns_bytes_and_flag(self):
        stencil = Image.new("1", (64, 64), 1)
        data, group4 = _encode_stencil(stencil)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)
        self.assertIsInstance(group4, bool)


if __name__ == "__main__":
    unittest.main()
