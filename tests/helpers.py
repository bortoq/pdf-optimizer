"""Synthetic PDFs shared by the test modules."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import shutil

import numpy as np
import pikepdf
from pikepdf import Dictionary, Name, Pdf, Stream
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, TiffImagePlugin

TEXT_LINE = "Pdfopt scanned text 12345"


def need(*tools: str) -> bool:
    return all(shutil.which(tool) for tool in tools)


def text_image(width=1000, height=1400, *, color=False, ink=(10, 10, 10),
               paper=(255, 255, 255), size=22, seed: int | None = None) -> Image.Image:
    pixels = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(pixels)
    font = ImageFont.load_default(size=size)
    rng = np.random.default_rng(seed)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz0123456789 "))
    for index, y in enumerate(range(70, height - 70, int(size * 1.8))):
        fill = ink
        if color and index % 3 == 1:
            fill = (190, 20, 20)
        line = TEXT_LINE if seed is None else "".join(rng.choice(letters, 28))
        draw.text((70, y), line, fill=fill, font=font)
    return pixels


def jpeg_bytes(image: Image.Image, quality=92, **options) -> bytes:
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=quality, **options)
    return buffer.getvalue()


def _page(pdf: Pdf, width: float, height: float, content: bytes,
          resources: Dictionary | None = None):
    page = pdf.add_blank_page(page_size=(width, height))
    stream = pdf.make_stream(content)
    page.Contents = stream
    page.Resources = resources or Dictionary()
    return page


def make_text_pdf(path: Path, lines: int = 5000) -> None:
    pdf = Pdf.new()
    _page(pdf, 612, 792, b"q\n" + b"0 0 m 100 100 l S\n" * lines + b"Q\n")
    pdf.save(path, compress_streams=False)


def image_stream(pdf: Pdf, data: bytes, width: int, height: int, *, filter_name,
                 colorspace=Name.DeviceRGB, bits=8, parms=None, **extra) -> Stream:
    stream = Stream(pdf, b"")
    stream.write(data, filter=filter_name, decode_parms=parms)
    stream.Type, stream.Subtype = Name.XObject, Name.Image
    stream.Width, stream.Height = width, height
    stream.BitsPerComponent = bits
    if colorspace is not None:
        stream.ColorSpace = colorspace
    for key, value in extra.items():
        stream["/" + key] = value
    return pdf.make_indirect(stream)


def make_scan_pdf(path: Path, *, pages: int = 1, color_text: bool = False,
                  gray: bool = False, optimize: bool = True, ocr: bool = True,
                  text_page: bool = True, flate: bool = False,
                  varied: bool = False) -> None:
    """Scan-like pages: one big image plus an invisible OCR text layer."""
    pdf = Pdf.new()
    font = pdf.make_indirect(Dictionary(Type=Name.Font, Subtype=Name.Type1,
                                        BaseFont=Name.Helvetica))
    for number in range(pages):
        if text_page:
            pixels = text_image(color=color_text,
                                seed=number + 1 if varied else None)
        else:
            pixels = Image.new("RGB", (400, 500))
            pixels.putdata([((x // 8) % 256, (y // 8) % 256, ((x + y) // 8) % 256)
                            for y in range(500) for x in range(400)])
        if gray:
            pixels = pixels.convert("L")
        space = Name.DeviceGray if gray else Name.DeviceRGB
        if flate:
            import zlib
            image = image_stream(pdf, zlib.compress(pixels.tobytes(), 6),
                                 pixels.width, pixels.height,
                                 filter_name=Name.FlateDecode, colorspace=space)
        else:
            image = image_stream(pdf, jpeg_bytes(pixels, optimize=optimize),
                                 pixels.width, pixels.height,
                                 filter_name=Name.DCTDecode, colorspace=space)
        content = b"q 360 0 0 504 0 0 cm /Im0 Do Q"
        if ocr:
            content += b" BT /F1 12 Tf 50 50 Td (OCR text) Tj ET"
        _page(pdf, 360, 504, content, Dictionary(
            XObject=Dictionary(Im0=image), Font=Dictionary(F1=font)))
    pdf.save(path)


def make_soft_scan_pdf(path: Path, *, blur: float = 1.0, noise: float = 2.5) -> None:
    """A 300 dpi grayscale scan with soft glyph edges and sensor noise: a
    bilevel stencil cannot reach SSIM 0.96 on it, but stays near 0.95."""
    rng = np.random.default_rng(1)
    image = text_image(1700, 2200, size=30, seed=1).convert("L")
    pixels = np.asarray(image.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32)
    pixels = pixels * 0.92 + rng.normal(0, noise, pixels.shape)
    data = jpeg_bytes(Image.fromarray(pixels.clip(0, 255).astype(np.uint8)), quality=90)
    pdf = Pdf.new()
    picture = image_stream(pdf, data, 1700, 2200, filter_name=Name.DCTDecode,
                           colorspace=Name.DeviceGray)
    _page(pdf, 408, 528, b"q 408 0 0 528 0 0 cm /Im0 Do Q",
          Dictionary(XObject=Dictionary(Im0=picture)))
    pdf.save(path)


def stripes(width=300, height=300, *, gray=False) -> np.ndarray:
    base = np.add.outer(np.arange(height), np.arange(width)) % 256
    base = base.astype(np.uint8)
    if gray:
        return base
    return np.stack([base, base[::-1], (base // 2).astype(np.uint8)], axis=2)


def make_flate_image_pdf(path: Path, array: np.ndarray, *, colorspace=None,
                         bits: int = 8, level: int = 1, **extra) -> None:
    import zlib
    height, width = array.shape[:2]
    if colorspace is None:
        colorspace = Name.DeviceGray if array.ndim == 2 else Name.DeviceRGB
    pdf = Pdf.new()
    image = image_stream(pdf, zlib.compress(array.tobytes(), level), width,
                         height, filter_name=Name.FlateDecode,
                         colorspace=colorspace, bits=bits, **extra)
    _page(pdf, 200, 200, b"q 200 0 0 200 0 0 cm /Im0 Do Q",
          Dictionary(XObject=Dictionary(Im0=image)))
    pdf.save(path)


def bilevel_bitmap(width=1200, height=1600) -> Image.Image:
    bitmap = Image.new("1", (width, height), 1)
    draw = ImageDraw.Draw(bitmap)
    font = ImageFont.load_default(size=26)
    rng = np.random.default_rng(7)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz0123456789 "))
    for y in range(60, height - 60, 46):
        draw.text((60, y), "".join(rng.choice(letters, 38)), fill=0, font=font)
    return bitmap


def g4_stream(bitmap: Image.Image) -> bytes:
    """A PDF-conformant Group 4 stream (black runs = black ink).

    Pillow's TIFF writer codes MinIsBlack ones as black runs, so the bitmap is
    inverted first.
    """
    bitmap = ImageOps.invert(bitmap.convert("L")).convert("1")
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[278] = bitmap.height
    output = BytesIO()
    bitmap.save(output, "TIFF", compression="group4", tiffinfo=tags)
    output.seek(0)
    with Image.open(output) as tiff:
        offset, count = tiff.tag_v2[273][0], tiff.tag_v2[279][0]
    return output.getvalue()[offset:offset + count]


def make_bilevel_pdf(path: Path, *, kind: str = "flate", mask: bool = False) -> None:
    import zlib
    bitmap = bilevel_bitmap()
    pdf = Pdf.new()
    extra = {"ImageMask": True} if mask else {}
    colorspace = None if mask else Name.DeviceGray
    if kind == "g4":
        image = image_stream(
            pdf, g4_stream(bitmap), bitmap.width, bitmap.height,
            filter_name=Name.CCITTFaxDecode, colorspace=colorspace, bits=1,
            parms=Dictionary(K=-1, Columns=bitmap.width, Rows=bitmap.height),
            **extra)
    else:
        image = image_stream(pdf, zlib.compress(bitmap.tobytes(), 6), bitmap.width,
                             bitmap.height, filter_name=Name.FlateDecode,
                             colorspace=colorspace, bits=1, **extra)
    _page(pdf, 300, 400, b"0 g q 300 0 0 400 0 0 cm /Im0 Do Q",
          Dictionary(XObject=Dictionary(Im0=image)))
    pdf.save(path)


def sign(path: Path) -> None:
    """Add a (fake but structurally real) signature dictionary."""
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        signature = pdf.make_indirect(Dictionary(
            Type=Name.Sig, Filter=Name.Adobe_PPKLite,
            ByteRange=[0, 1, 2, 3], Contents=pikepdf.String(b"00")))
        field = pdf.make_indirect(Dictionary(
            FT=Name.Sig, T=pikepdf.String("Sig1"), V=signature))
        pdf.Root.AcroForm = Dictionary(Fields=[field], SigFlags=3)
        pdf.save(path)
