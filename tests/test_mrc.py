from PIL import Image
from pypdf.generic import ArrayObject, NameObject, NumberObject
import unittest

from pdfopt.mrc import _eligible, _encode_stencil, _image_stream


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


class StencilEncodingTests(unittest.TestCase):
    def test_returns_bytes_and_flag(self):
        stencil = Image.new("1", (64, 64), 1)
        data, group4 = _encode_stencil(stencil)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)
        self.assertIsInstance(group4, bool)


if __name__ == "__main__":
    unittest.main()
