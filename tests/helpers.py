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
                  varied: bool = False, icc: bool = False) -> None:
    """Scan-like pages: one big image plus an invisible OCR text layer.

    ``icc`` tags the (colour) images with an ICCBased sRGB colour space.
    """
    pdf = Pdf.new()
    font = pdf.make_indirect(Dictionary(Type=Name.Font, Subtype=Name.Type1,
                                        BaseFont=Name.Helvetica))
    icc_space = None
    if icc:
        from PIL import ImageCms
        profile = Stream(pdf, ImageCms.ImageCmsProfile(
            ImageCms.createProfile("sRGB")).tobytes())
        profile.N = 3
        icc_space = pikepdf.Array([Name.ICCBased, pdf.make_indirect(profile)])
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
        space = Name.DeviceGray if gray else (icc_space or Name.DeviceRGB)
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


def truetype_font(glyphs: int = 200) -> bytes:
    """A TrueType font with ``glyphs`` distinct, detailed glyphs mapped from
    U+0041 on (so cutting unused glyphs is measurable)."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    names = [".notdef"] + [f"g{i:03d}" for i in range(glyphs)]
    outlines = {}
    for index, name in enumerate(names):
        pen = TTGlyphPen(None)
        # A star-like polygon whose shape depends on the glyph index.
        points = []
        for k in range(24):
            radius = 300 if k % 2 else 150 + (index * 37 + k * 11) % 140
            angle = 2 * np.pi * k / 24 + index * 0.01
            points.append((int(350 + radius * np.cos(angle)), int(350 + radius * np.sin(angle))))
        pen.moveTo(points[0])
        for point in points[1:]:
            pen.lineTo(point)
        pen.closePath()
        outlines[name] = pen.glyph()
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({0x41 + i: f"g{i:03d}" for i in range(glyphs)})
    builder.setupGlyf(outlines)
    builder.setupHorizontalMetrics(dict.fromkeys(names, (700, 50)))
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "PdfoptTest", "styleName": "Regular"})
    builder.setupOS2()
    builder.setupPost()
    output = BytesIO()
    builder.save(output)
    return output.getvalue()


def make_font_pdf(path: Path, *, kind: str = "simple", text: str = "ABAC",
                  in_form: bool = False, annotation: bool = False, acroform: bool = False,
                  orphan_user: bool = False, encoding: str = "/Identity-H") -> None:
    """A page showing ``text`` with an embedded TrueType font.

    kind: "simple" (WinAnsi TrueType) or "cid" (Type0/CIDFontType2, codes =
    glyph ids). ``in_form`` draws the text inside a form XObject,
    ``annotation`` (also) in an annotation appearance, ``orphan_user`` adds an
    unreachable font dictionary sharing the program.
    """
    import zlib
    pdf = Pdf.new()
    data = truetype_font()
    program = pdf.make_stream(zlib.compress(data, 9))
    program.Filter, program.Length1 = Name.FlateDecode, len(data)
    descriptor = pdf.make_indirect(Dictionary(
        Type=Name.FontDescriptor, FontName=Name("/PdfoptTest"), Flags=32,
        FontBBox=[0, -200, 1000, 800], ItalicAngle=0, Ascent=800, Descent=-200,
        CapHeight=700, StemV=80, FontFile2=program))
    if kind == "simple":
        font = Dictionary(Type=Name.Font, Subtype=Name.TrueType, BaseFont=Name("/PdfoptTest"),
                          FirstChar=32, LastChar=255, Widths=[700] * 224,
                          Encoding=Name.WinAnsiEncoding, FontDescriptor=descriptor)
        shown = pikepdf.String(text.encode("cp1252"))
    else:
        cid = Dictionary(Type=Name.Font, Subtype=Name.CIDFontType2,
                         BaseFont=Name("/PdfoptTest"), FontDescriptor=descriptor,
                         CIDSystemInfo=Dictionary(Registry="Adobe", Ordering="Identity",
                                                  Supplement=0),
                         CIDToGIDMap=Name.Identity, DW=700)
        font = Dictionary(Type=Name.Font, Subtype=Name.Type0, BaseFont=Name("/PdfoptTest"),
                          Encoding=Name(encoding), DescendantFonts=[pdf.make_indirect(cid)])
        # glyph id of the k-th letter is k + 1 (after .notdef)
        shown = pikepdf.String(b"".join((ord(c) - 0x41 + 1).to_bytes(2, "big") for c in text))
    font = pdf.make_indirect(font)
    if orphan_user:
        pdf.Root.Orphan = pdf.make_indirect(Dictionary(
            Type=Name.Font, Subtype=Name.TrueType, BaseFont=Name("/PdfoptTest"),
            Encoding=Name.WinAnsiEncoding, FontDescriptor=descriptor))
    text_ops = b"BT /F1 48 Tf 40 100 Td " + shown.unparse() + b" Tj ET"
    resources = Dictionary(Font=Dictionary(F1=font))
    if in_form:
        form = pdf.make_stream(text_ops)
        form.Type, form.Subtype, form.BBox = Name.XObject, Name.Form, [0, 0, 400, 200]
        form.Resources = resources
        page = _page(pdf, 400, 200, b"q /Fm0 Do Q",
                     Dictionary(XObject=Dictionary(Fm0=form)))
    else:
        page = _page(pdf, 400, 200, text_ops, resources)
    if annotation:
        appearance = pdf.make_stream(b"BT /F1 24 Tf 10 10 Td "
                                     + (pikepdf.String(b"Z").unparse() if kind == "simple"
                                        else pikepdf.String((26).to_bytes(2, "big")).unparse())
                                     + b" Tj ET")
        appearance.Type, appearance.Subtype = Name.XObject, Name.Form
        appearance.BBox, appearance.Resources = [0, 0, 100, 50], resources
        page.Annots = pdf.make_indirect(pikepdf.Array([pdf.make_indirect(Dictionary(
            Type=Name.Annot, Subtype=Name.Stamp, Rect=[250, 20, 350, 70],
            AP=Dictionary(N=appearance)))]))
    if acroform:
        pdf.Root.AcroForm = Dictionary(Fields=[], DR=resources)
    pdf.save(path)
