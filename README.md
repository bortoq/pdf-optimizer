# pdfopt

A PDF optimizer. It tries several encoders, picks the smallest **verified**
result and replaces the source file atomically, only if the file got smaller.

- **Lossless (default)**: text and images stay as they are; every page whose
  images or fonts changed is compared pixel for pixel at 100 dpi.
- **Lossy (`-q N`)**: recompresses photos and scans for as long as the SSIM of
  every page stays at or above `N/100`.

Everything runs locally; no file leaves your machine. Design notes are in
[docs/architecture.md](docs/architecture.md).

## Results

Scans at 300 dpi, 9 pages, page SSIM ≥ 0.90 (`-q 90`), against Ghostscript:

| File | Source | pdfopt `-q 90` | Ghostscript `/ebook` | Ghostscript `/screen` |
| --- | --- | --- | --- | --- |
| grayscale scan | 14,939 KB | **104 KB**, SSIM 0.965, 11 s | 1,719 KB, SSIM 0.958 | 685 KB, SSIM 0.889 |
| colour scan | 12,916 KB | **112 KB**, SSIM 0.970, 22 s | 11,200 KB | 1,936 KB, SSIM 0.896 |

Lossless (KB, source → pdfopt; `pikepdf`/qpdf in brackets):

| Document | Result |
| --- | --- |
| dvips paper | 146 → **124** (130) |
| pdfTeX paper | 776 → **570** (747) |
| Word document | 1595 → **1073** (1532) |
| CCITT scan | 553 → **396** (539) |

An independent audit measured a 6-page 300 dpi scan (7.8 MB) at 111 KB with
SSIM 0.962, against 655 KB / 0.880 for DjVu (`pdf2djvu --lossy`).

## Install

```sh
pip install -e .            # pikepdf, pypdf, Pillow, numpy, fontTools, mozjpeg + builds ECT
pip install -e '.[dev]'     # + pytest, ruff
```

Python ≥ 3.11 and C/C++ compilers are needed (the ECT module is built on
install; if the build fails, installation still works and streams are packed
with zlib). Running straight from a source checkout: `python3 setup.py build_ext --inplace`.
External programs are listed [below](#external-programs).

**Docker** (all programs included):

```sh
docker build -t pdfopt .
docker run --rm -v "$PWD":/data pdfopt -q 96 book.pdf
```

## Usage

```sh
pdfopt book.pdf                      # lossless
pdfopt -q 96 book.pdf                # lossy, page SSIM >= 0.96
pdfopt --dry-run -r /path/to/books   # estimate the saving for a tree
pdfopt -v *.pdf                      # verbose log
pdfopt --decrypt locked.pdf          # encrypted, but opens without a password
pdfopt -o small.pdf book.pdf         # write a new file, keep the source
pdfopt -o out/ -j 4 *.pdf            # into out/, at most 4 cores
python3 -m pdfopt --help
```

| Option | Meaning |
| --- | --- |
| `-q`, `--quality N` | minimum page SSIM `N/100` (1–100); turns on lossy mode |
| `--dry-run` | compute the result without replacing the source |
| `-r`, `--recursive` | descend into subfolders |
| `-v`, `--verbose` | per-file results; on stderr the size and rejection reason of each engine, timings, warnings about missing programs |
| `--decrypt` | decrypt files that open without a password and optimize them (the owner's restrictions, such as printing or copying, are dropped) |
| `-o`, `--output PATH` | leave sources alone: write the result to a file (one input) or a directory (several); unchanged and skipped files are copied as they are |
| `-j`, `--jobs N` | cores to use (default: all) |
| `--version` | print the version |
| `--force` | process a file even if it is already optimized |
| `--safe-text` | with `-q`, code text masks without JBIG2 symbol mode (larger, but similar letters can never be swapped) |
| `--mrc-quality N` | minimum SSIM of MRC text pages (default 6 below `-q`; 100 with `-q 100`) |

**Exit codes:** `0` success (including "no change" and skipped files), `1` at
least one file failed, `2` bad arguments, or nothing to do because every path
named was rejected (a symlink, not a `.pdf`).

Without `-v` a terminal shows one progress line with the share saved so far,
and a single total at the end. Signed PDFs and password-protected files are
**skipped** (`skipped: ...`), which is not an error; files encrypted without a
password are processed only with `--decrypt`. Already optimized files are
counted in the total. When several files run at once, their number is bounded
by free memory (about 700 MB per file with `-q`).

## What it does

### Lossless

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

### Lossy (`-q`)

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

## External programs

| Program | Used for | Without it |
| --- | --- | --- |
| `pdftoppm` (poppler) | render checks, **required for `-q`** | lossless drops image transforms; `-q` unavailable |
| `jbig2` (jbig2enc) | JBIG2 generic and symbol dictionaries | Group 4/Flate |
| `opj_compress` (OpenJPEG) | JPEG 2000 backgrounds | JPEG only |
| `cjpeg` from mozjpeg | denser JPEG | Pillow/libjpeg |
| `gs` (Ghostscript) | Ghostscript profiles | skipped |
| `hpdf-worker` (in `PATH`) | Hyper Compress | skipped with a warning |

## Layout

```
pdfopt/
  cli.py        arguments, file discovery, parallelism across files
  optimize.py   orchestrator: stages, candidates, verification, marker, atomic replace
  lossless.py   lossless passes (structure, images, JPEG, ECT, save)
  fonts.py      TrueType subsetting
  jpegopt.py    lossy photo recompression
  mrc.py        layered scan compression
  pdfmodel.py   reading: signature, page fingerprint, marker, image walk
  imagedata.py  image decoding, PNG predictors, Group 4
  render.py     pdftoppm, VisualGate (SSIM), identity check
  ssim.py       SSIM matching ffmpeg
  tools.py      external program lookup
```

## Tests

```sh
pytest
ruff check .
```

CI runs the tests on Linux, macOS and Windows. Tests that need `pdftoppm`,
`jbig2`, `ffmpeg`, `gs` or OpenJPEG are skipped when the program is missing.

## License

MIT, see [LICENSE](LICENSE). The bundled ECT and Zopfli (`pdfopt/_ect`) are under
the Apache License 2.0, see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
