# Changelog

## Unreleased

**Compression**
- Type 1 fonts (pdfTeX, dvips) are repacked as CFF (`/FontFile3`, `Type1C`)
  with the same outlines, widths, hints, flex and encoding. Their glyphs are
  stored encrypted, which Flate cannot compress: papers shrink by half or more
  (−2.6 → −57.7 %, −6.8 → −72.2 %) and lossless. Pages showing a repacked font
  are compared pixel for pixel like every other change.
- When the rendering check fails, only the fonts shown on the failing pages are
  left alone (`lossless-fewer-fonts`); before, every font change was dropped.
- An empty AcroForm (hyperref writes one into most papers) no longer stops the
  font step; with form fields fonts are still not cut, but Type 1 fonts are
  repacked (no glyph goes).
- Lossless palettes: pictures with up to 256 RGB colours or 16 gray levels are
  stored as indices into a colour table (`/Indexed`, 1–8 bits).
- Object streams and the cross-reference stream of the result are packed with
  ECT (qpdf uses zlib): 0.2–3 % of a file.
- `-q`: gray Flate photos become JPEG (only colour ones did; a gray picture has
  at most 256 levels, below the photo test of 1500 colours). Screenshots and
  diagrams get the smallest palette (256…4 colours) that keeps the SSIM instead
  of staying as they were. On a set of books and papers: −22.6 → −51.7 %,
  −18.1 → −29.3 %, −50.4 → −86.9 % per file, lowest page SSIM unchanged.
- On a set of 35 books and papers the lossless saving went from 25.2 % to 27.0 %.

**Stopping a run (found by repeating `kill` under full CPU load)**
- A termination request while worker processes were starting could leave a
  worker that ignored all further requests, and the run never ended. Stop
  signals are now held while workers start and released in each worker once
  its handler is set.
- MRC page processes (inside a worker) were not told when the run was
  terminated, so a worker waited for them to finish their pages. They are
  told now, and end after removing their page folder (their `opj_compress`
  is killed with them).
- A worker that had been stopped could take the next queued file (queued
  tasks cannot be cancelled) or wait for ever on a queue lock held by a
  stopped worker; two requests sent at once arrive as one signal. A stopped
  worker now starts no other file and ends after its answer.
- A work folder made at the moment of the interruption could stay behind
  (`TemporaryDirectory` creates the folder before it can remove it). Work
  folders are now named before they are made and removed after an
  interruption in any case.
- After an interruption the process ends without finalizing the interpreter,
  which could crash in native code cut off in the middle (exit 130 instead of
  a segmentation fault).

**Audit fixes (second round)**
- A damaged file can no longer stop a run. A page tree whose `/Parent` links form
  a loop made pdfopt hang for ever; a wrong object type in a page (`/Rotate` as a
  string, `/Contents` as a dictionary...) ended the whole batch with a traceback.
  Both are now an `error:` line for that file, and the other files are processed.
- MRC ink palette: seeding a new ink compared every far cell with 2000 others,
  which on a page of two inks in equal amounts (60 000 cells) took 4.6 GB and
  11 s in each page process. The seed is now chosen within a sample of 2000:
  0.4 s and 160 MB, same palette.
- PDF/A-1 stays PDF/A-1: a file whose XMP claims part 1 keeps its PDF version and
  gets no object or xref streams and no JPEG 2000 backgrounds (all three are
  forbidden there); Ghostscript is not offered to PDF/A files of any part.
- `--dry-run` works in a folder without write access (the work folder then goes
  to the system's temporary folder).
- The new file is flushed to the disk (`fsync`) before it replaces the source,
  and the folder after: a power cut leaves the old file or the new one.
- `SIGTERM` (`kill`, `docker stop`, `timeout`) ends a run like Ctrl+C: worker
  processes are told, work folders are removed, sources stay unchanged, exit
  code 130.
- `-o`: the result is written under a temporary name and renamed when complete.
  An output file that existed before a failed or interrupted run is left as it
  was (it used to be overwritten and then deleted).
- `-r` does not descend into work folders (`.pdfopt-*`) left by a killed run.
- A candidate from an engine that rewrites the whole file (Ghostscript, Hyper
  Compress) is rejected when the document lost its outlines, page labels,
  language, viewer preferences, output intents, structure tree, layers, page
  mode/layout or open action. After Ghostscript the language, viewer
  preferences, page mode/layout and the Info entries of the source are put back
  (it dropped the first ones and wrote its own Producer and dates).

**Interface**
- Files that gained nothing are remembered (`~/.cache/pdfopt/no-gain.json`, or
  `$XDG_CACHE_HOME`, `%LOCALAPPDATA%` on Windows) by content and settings, so the
  next run over a library does not analyse them again; `-v` shows `no change
  (known from an earlier run)`. `--force`, another `-q`, a new pdfopt version or
  a newly installed program make them be analysed again.
- The marker records the `-q` a lossy file was made with. A lower `-q` processes
  the file again; the same or a higher one still answers "already optimized"
  (as do files marked before this change).

**Dependencies and build**
- `pypdf` is no longer needed: the text of Ghostscript/Hyper candidates is
  compared with poppler's `pdftotext`, which comes with `pdftoppm`.
- The version is defined once (`pdfopt.__version__`).
- CI: Python 3.11–3.14 on Linux, a weekly run against new releases of the
  dependencies, `mypy` (the package is typed, `py.typed`), deprecation warnings
  are shown again. A failed build of the ECT module fails CI and the release
  (`PDFOPT_REQUIRE_ECT=1`) instead of silently giving a wheel without it.
- Release: the tests run before publishing; wheels also for Python 3.14, Linux
  aarch64 and Intel macOS; every wheel is checked to contain a working ECT
  module.

**Interface**
- The progress bar moves while a file is being processed (layering of scan
  pages, rendering checks, ECT), not only when a file is finished. It shows the
  share of the total work (weighted by file size), files done and the saving so
  far. Works with several files at once (worker processes report to the parent).
- With `-q` the total line shows the similarity of the new files to the sources:
  `Saved: 281.0 MB (87.1%) SSIM 0.9983 (lowest page 0.9712)`. The mean is over
  all pages of all changed files; each page is rendered once more at 200 dpi from
  the file as it is written. `optimize_pdf` returns an `Outcome` (still unpacks as
  `(status, saved)`) with the page `scores`.

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
