# How it works

## Lossless

One parse, several passes, one save:

1. drops page thumbnails, merges identical streams;
2. images: RGB with R=G=B becomes DeviceGray, pure black-and-white 8-bit images
   become 1-bit, Flate images get the best PNG predictor, 1-bit Flate and CCITT
   Group 4 are rewritten as **JBIG2 generic** (lossless);
3. JPEG: mozjpeg's `jpegtran` (optimized Huffman tables, tuned progressive
   scans) with a pixel-identity check, 0.7–28 % smaller than libjpeg's;
4. embedded TrueType fonts are cut to the glyphs the document shows (fontTools,
   glyph ids kept, content streams unchanged); the cmap tables are rebuilt, so
   fonts without `/Encoding` (LibreOffice) work too. Pages with changed fonts
   are compared pixel for pixel; with an AcroForm or any unclear font use the
   font is left alone, and if a cut ever changes the rendering only the font
   step is dropped, not the image steps;
5. streams are recompressed by the bundled ECT (like `ect -9`; streams above
   4 MB like `ect -6`) and saved as object/xref streams.

Metadata, text layer, geometry and annotations do not change.

## Lossy (`-q`)

On top of the lossless result:

- **MRC** for scans (page = one picture): a text-free background at 50–100 dpi
  (JPEG or JPEG 2000, quality searched by SSIM), a full-resolution text mask
  (JBIG2 with shared dictionaries, Group 4 or Flate), one ink colour per mask.
  The JBIG2 symbol threshold depends on letter size, and very small text is
  coded losslessly, so 6 and 8 are never merged the way Xerox scanners did in 2013.
- **JPEG** for other photos: the smallest JPEG (and dpi, from the content-stream
  geometry) that passes SSIM; Flate photos become JPEG.
- **Ghostscript** profiles (300/240/200 dpi) for PDFs without text or forms.
- **Hyper Compress**, an external `hpdf-worker`, if found in `PATH`.

## How a result is verified

Candidates are sorted by size and checked from the smallest; the first that
passes wins.

| Check | When |
| --- | --- |
| page count, MediaBox/CropBox, rotation, annotation count | always |
| content stream of every page byte for byte (else text compare) | lossless |
| metadata (Info and XMP) | lossless |
| rendering identical at 100 dpi on changed pages | lossless, when images or fonts changed |
| rendering SSIM (200 dpi, `pdftoppm`) ≥ threshold | lossy: every MRC page and re-encoded page (failing pages revert), all pages for Ghostscript/Hyper |

SSIM is computed in-process (numpy) with FFmpeg's `ssim` formula.

A marker (`/Root /PdfOpt`) keeps a second run from recompressing a file; `--force`
overrides it.
