from pathlib import Path
import tempfile
import unittest

import pikepdf

from pdfopt.pdfmodel import (
    Fingerprint,
    Skipped,
    collect_images,
    compare,
    has_signature,
    open_pdf,
    read_marker,
    set_marker,
)

from .helpers import make_scan_pdf, make_text_pdf, sign


class SafetyTests(unittest.TestCase):
    def test_real_signature_is_detected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "signed.pdf"
            make_text_pdf(path)
            sign(path)
            with open_pdf(path) as pdf:
                self.assertTrue(has_signature(pdf))

    def test_text_mentioning_byterange_is_not_a_signature(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "book.pdf"
            make_text_pdf(path)
            with path.open("ab") as stream:
                stream.write(b"\n% /ByteRange [0 1 2 3]\n")
            with open_pdf(path) as pdf:
                self.assertFalse(has_signature(pdf))

    def test_encrypted_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "locked.pdf"
            make_text_pdf(path)
            with pikepdf.open(path, allow_overwriting_input=True) as pdf:
                pdf.save(path, encryption=pikepdf.Encryption(owner="o", user="u"))
            with self.assertRaises(Skipped):
                open_pdf(path)

    def test_garbage_is_a_regular_error(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bad.pdf"
            path.write_bytes(b"not a pdf")
            with self.assertRaises(ValueError) as caught:
                open_pdf(path)
            self.assertNotIsInstance(caught.exception, Skipped)


class FingerprintTests(unittest.TestCase):
    def fingerprint(self, path):
        with open_pdf(path) as pdf:
            return Fingerprint.read(pdf, path)

    def test_fingerprint_ignores_stream_compression(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = Path(root) / "a.pdf", Path(root) / "b.pdf"
            make_text_pdf(first)
            with pikepdf.open(first) as pdf:
                pdf.save(second, compress_streams=False)
            compare(self.fingerprint(second), self.fingerprint(first),
                    content=True, metadata=True)

    def test_changed_content_is_rejected_when_content_must_match(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = Path(root) / "a.pdf", Path(root) / "b.pdf"
            make_text_pdf(first, lines=10)
            make_text_pdf(second, lines=11)
            with self.assertRaises(ValueError):
                compare(self.fingerprint(second), self.fingerprint(first),
                        content=True, metadata=False)

    def test_changed_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = Path(root) / "a.pdf", Path(root) / "b.pdf"
            make_text_pdf(first)
            with pikepdf.open(first) as pdf:
                pdf.pages[0].obj.Rotate = 90
                pdf.save(second)
            with self.assertRaises(ValueError):
                compare(self.fingerprint(second), self.fingerprint(first),
                        content=False, metadata=False)

    def test_changed_metadata_is_rejected_only_when_requested(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = Path(root) / "a.pdf", Path(root) / "b.pdf"
            make_text_pdf(first)
            with pikepdf.open(first) as pdf:
                pdf.docinfo["/Title"] = "changed"
                pdf.save(second)
            compare(self.fingerprint(second), self.fingerprint(first),
                    content=True, metadata=False)
            with self.assertRaises(ValueError):
                compare(self.fingerprint(second), self.fingerprint(first),
                        content=True, metadata=True)

    def test_text_layer_and_interactivity_flags(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "scan.pdf"
            make_scan_pdf(path)
            facts = self.fingerprint(path)
            self.assertTrue(facts.has_text_layer)
            self.assertFalse(facts.interactive)


class MarkerTests(unittest.TestCase):
    def test_marker_round_trip_and_sticky_lossy_flag(self):
        pdf = pikepdf.Pdf.new()
        self.assertIsNone(read_marker(pdf))
        set_marker(pdf, lossy=True)
        set_marker(pdf, lossy=False)
        version, lossy = read_marker(pdf)
        self.assertGreaterEqual(version, 1)
        self.assertTrue(lossy)


class ImageCollectionTests(unittest.TestCase):
    def test_images_know_their_pages(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "scan.pdf"
            make_scan_pdf(path, pages=2, text_page=False)
            with open_pdf(path) as pdf:
                images = collect_images(pdf)
            self.assertEqual(len(images), 2)
            self.assertEqual(sorted(min(ref.pages) for ref in images.values()), [0, 1])


if __name__ == "__main__":
    unittest.main()
