# Changelog

## Unreleased

**Audit fixes**
- Font cutting: the cmap tables are rebuilt after the subset. Fonts without
  `/Encoding` and with a single (1, 0) table (LibreOffice export) lost the way
  from code to glyph, the rendering check rejected the result, and the image
  changes were switched off together with the fonts. This works now, and when a
  font cut fails, only the font cutting is dropped (`lossless-nofonts`).
- `-o`/`--output` (a file or a folder; the sources are not touched),
  `-j`/`--jobs`, `--version`. A symlink or a non-PDF named on the command line
  is explained (exit code 2 when there is nothing to do); the total shows
  "already optimized: N".
- The number of files processed at once is bounded by free memory (about 700 MB
  per file with `-q`, 300 MB without).
- Dockerfile with all external programs; English README and a results table;
  release workflow (cibuildwheel wheels + sdist → PyPI on a `v*` tag);
  `py.typed`.
- The "pixel-identical" wording is exact now: the check runs at 100 dpi.
- All messages, `--help`, docs and tests are in plain English.

**Speed**
- The render check splits pages into chunks by core count instead of 8 pages:
  with 9 pages only 2 of 12 threads were busy. Layered (MRC) mode: a 9-page
  colour scan 55 → 22 s, a grayscale one 42 → 11 s (same result).
- Compositing while the background is searched is about 15 % faster.

**Interface**
- Without `-v` the output is minimal: a progress line with the share saved (on a
  terminal) and a total line. Warnings about `hpdf-worker` and ECT appear only
  with `-v`; fontTools warnings about slightly damaged fonts are silenced.
- `--decrypt`: files with encryption that open without a password are decrypted
  and optimized (the result has no encryption). Files that need a password are
  still skipped.

**Lossless**
- TrueType fonts are cut to the glyphs in use (fontTools, glyph ids kept): a Word
  document 1414 → 1073 KB (−24 %), pdfTeX with images from Office −16 %.
- JPEG is repacked with mozjpeg's `jpegtran` instead of libjpeg: −0.7 % on photo
  scans, −12…28 % on JPEG in papers (files −6…8 %); the same for the JPEG that
  pdfopt creates itself.
- ECT level 6 for streams above 4 MB: a paper with a 25 MB image 38 s → 11 s at
  a cost of about 0.1 %; Flate is not unpacked beyond the ECT budget.
- The ECT module rejects levels outside 2–9 (level 1 did not finish).
- New required dependencies: `fonttools`, `mozjpeg-lossless-optimization`;
  `jpegtran` is no longer needed.

**Lossy**
- MRC: size on par with DjVu at the same quality (9-page scans, `-q 90` and
  `-q 96`: grayscale 298 → 104 KB, colour 390 → 112 KB; DjVu 108 and 111 KB).
  - JBIG2 with a symbol match threshold chosen by letter size, per band of
    lines: 0.80 for normal text, 0.85/0.90 for small text, very small text
    lossless; footnotes get their own mask (protects against 6↔8 and 9↔8 swaps).
  - Thresholds stay at 0.9 or below: jbig2enc 0.28 (Ubuntu 24.04) rejected
    0.92, and the symbol mode silently did not work.
  - The mask is smoothed by a 3×3 vote.
  - Background at 100/50 dpi instead of 150/100; JPEG 2000 with ratios up to
    2000:1 (before, at most 50:1, which almost never beat JPEG).
  - Coloured text: one mask per ink instead of a colour JPEG image.
  - A separate threshold for text pages, `--mrc-quality` (default `-q` − 6):
    with `-q 96` MRC used to fail on real scans (a 10.9 MB grayscale scan →
    104 KB).
  - Scans with an ICC profile (ICCBased, CalRGB) keep their colour space in the
    layers; before, the background was written as DeviceRGB/DeviceGray.
- Ghostscript is really offered to documents without text: the text-layer test
  fired on any page with content, so the engine never ran.

**Reliability**
- A failure on one MRC page no longer cancels MRC and JPEG for the whole
  document.
- Image decoding is bounded (512 MB; Flate only to the needed size): a
  decompression bomb cannot exhaust memory. Pillow's `DecompressionBombError`
  is caught everywhere.
- Lossless RGB→Gray leaves images with `/Decode` alone (the array would become
  invalid).
- When root replaces a file, the owner is kept.

**Build**
- MIT license (`LICENSE`); the package as a whole is `MIT AND Apache-2.0`
  because of the bundled ECT and Zopfli.
- The sdist builds: `MANIFEST.in` with the ECT headers; the ECT/Zopfli licenses
  (Apache 2.0) go into the sdist and the wheel (`license-files`); the C++
  sources are no longer put in the wheel. CI builds the sdist and a wheel from it.
- CI compares SSIM with ffmpeg again and runs the Ghostscript tests.
- CI runs on Linux, macOS and Windows; `setup.py` knows MSVC and the ECT
  extension is optional (zlib is used if it does not build).

**API**
- `optimize_pdf`: everything after `mode` is keyword-only (a positional
  `dry_run` was easily mixed up with the threshold).

## TODO

- **Smoother `-q` scale on scans (decision postponed).** `-q 90`, `-q 85` and
  `-q 80` give byte-identical files on scans, while `-q 96` gives a file four
  times larger. The search reaches its floor (background at the lowest dpi and
  JPEG quality). Options: lower the floors (more dpi steps, a wider JPEG quality
  range) and measure the quality cost, or print the threshold that was really
  reached.

## 0.2.0

**Lossless**
- One parse of the file instead of five independent engines; the passes add up.
- pikepdf instead of the external `qpdf` (object/xref streams are always
  available), ECT -9 (sources from katzip, `pdfopt/_ect`) for Deflate.
- Image re-encoding: RGB→Gray, 8→1 bit, PNG predictors, G4/Flate→JBIG2 generic.
- Stream deduplication, thumbnail removal.
- Pixel check of the changed pages.

**Lossy**
- MRC: adaptive binarization, normalized background filling, a colour text
  layer, empty pages, JBIG2 in chunks, scans in Flate/JPX, shared layers.
- JPEG: dpi from the page geometry, quality bisection, Flate photos → JPEG.

**Speed**
- SSIM in-process (ffmpeg's formula), no `ffmpeg` process per page.
- Candidates are checked from the smallest; the fingerprint uses content hashes
  instead of text extraction.
- Rendering in chunks, a shared reference, early exit; parallelism across files.

**Reliability**
- The `/PdfOpt` marker: no second compression; `--force`.
- Signed or encrypted files are "skipped", exit code 0.
- The signature is searched in objects, not in the file text.

**CLI:** `-v`, `--force`, `--safe-text`. Removed: `-j` (always all cores; it
came back later), `--fast` (always ECT -9), `--version` (it came back later),
`--pdf-worker` (`hpdf-worker` is taken from `PATH`, a warning if it is missing),
`--mode`, `--lossless`, `--no-dry-run`, `--no-recursive`, the variables
`*_HYPER_WORKER`/`HYPER_DRV`, the `third_party` folder. Fixed a crash of
`pdfopt -h` (a `%` sign in the help).
