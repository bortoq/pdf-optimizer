# pdfopt architecture

## Data flow

```
source ──► open (pikepdf) ─► signed/encrypted? ─► Skipped
              │
              ├─► Fingerprint (pages, content hashes, metadata)
              └─► /PdfOpt marker? ─► "already optimized"

lossless.build ──► [thumbnails, dedupe] ─► [images, JPEG] ─► [fonts] ─► [ECT] ─► save
      │                                                                    │
      └── check: Fingerprint + pixel-identical changed pages ──────────────┘
              │ failed → retry without font cutting, then without image changes
              ▼
         "lossless" stage (a candidate; in lossy mode also the base for the next ones)
              │
   -q ────────┼──► MRC ──► (pages passed SSIM) ──► + JPEG ──► candidates
              ├──► Ghostscript, 3 profiles
              └──► Hyper Compress

candidates by size ─► first one that passes the checks ─► marker ─► os.replace
```

Every stage works on one open `pikepdf.Pdf`: the file is parsed once and saved
once. The passes add up in one chain. (Older versions had separate engines:
`pypdf`, `qpdf`, `jpegtran`, `pypdf+qpdf`, `pypdf+jpegtran`.)

## Checks

Rule: **no candidate reaches the file without a check**, and only what could
really have changed is checked.

- `pdfmodel.Fingerprint` holds MediaBox/CropBox/Rotate, the annotation count and
  the SHA-256 of the decoded content streams of every page. The lossless passes
  and MRC/JPEG do not touch content streams, so the comparison is a fast byte
  comparison. Text extraction (`pypdf`, slow) is needed only when the content
  changed (Ghostscript, Hyper) and is computed lazily. The fingerprint also
  records whether a page has text: a `BT` operator in the page content or in its
  forms (unused fonts in the resources do not count; an unreadable stream counts
  as text). Ghostscript is offered only to documents without text.
- `render.identical_pages` checks that the 100 dpi rendering of pages with
  changed images or fonts is pixel-identical (RGB→Gray, 8→1 bit, JBIG2 generic,
  predictors, cut fonts). If it fails, the build is repeated without font
  cutting (if fonts were cut), then without image changes.
- `render.VisualGate` computes the SSIM of the 200 dpi rendering against a
  reference. The reference is rendered once (PNG, in parallel; a chunk is at
  most 8 pages and at most `ceil(pages / cores)`, otherwise 9 pages would use
  two threads of twelve) and shared by all candidates. A candidate is rendered
  in chunks and compared at once; the first bad page stops the check
  (`stop_early`).
- `ssim.ssim` uses 8×8 windows with a stride of 4, like `ffmpeg -filter ssim`,
  and agrees with ffmpeg to 6 digits (test `test_matches_ffmpeg`).

## Lossless passes (`lossless.py`)

| Pass | Idea | Guarantee |
| --- | --- | --- |
| thumbnails | a viewer rebuilds `/Thumb` | page metadata untouched |
| dedupe | streams with equal bytes and dictionary become one reference | identical data |
| images | RGB(R=G=B)→Gray, {0,255}→1 bit, PNG predictor, G4/Flate→JBIG2 generic | pixel-identical rendering |
| JPEG | mozjpeg's `jpegtran` (`mozjpeg-lossless-optimization`): Huffman tables and tuned progressive scans, markers are not copied; 0.7 % on photo scans, up to 28 % on smooth images beyond libjpeg's `jpegtran` | decoded pixels are compared |
| fonts (`fonts.py`) | embedded TrueType (FontFile2) is cut by fontTools to the glyphs whose codes are shown by the text operators of pages, forms, annotation appearances, tiling patterns, soft masks and Type 3 glyphs; `retain_gids`, tables and hinting stay; the cmap tables are rebuilt after the cut (fontTools keeps only the requested Unicode values and empties the (1, 0) table of LibreOffice fonts without `/Encoding`). Left alone: documents with an AcroForm, Type0 fonts with an encoding other than Identity, font programs used by a font dictionary outside the walked resources (or a direct dictionary), and everything when a content stream cannot be parsed | content streams byte for byte; pages with changed fonts are pixel-identical |
| ECT | recompresses Flate with the bundled compressor (`pdfopt/_ect`; it releases the GIL, so streams scale over cores): level 9, streams above 4 MB level 6 (`ect_mode`: 4–8 times faster, 0.03–0.3 % larger; a 25 MB image went from 38 s to about 5 s); a budget of 256 MB of unpacked data per document, Flate is never unpacked beyond the budget. ECT level 1 does not finish, so the module accepts only 2–9 | Deflate is lossless |
| save | object streams + xref streams (qpdf) | - |

Images are read in the main thread (qpdf objects are not thread safe) and
encoded in a thread pool, in windows of `2·jobs` images to bound memory.

## MRC (`mrc.py`)

1. `scan_candidates`: pages with exactly one image (8 bit, Gray/RGB, ≥500 px,
   no masks, no `/Decode`). Equal images on different pages are layered once
   and the layers are shared.
2. `ink_mask`: the paper is estimated from a reduced copy (max filter + blur);
   ink is every pixel darker than the paper by more than `max(30, 30 %)`; the
   polarity comes from the median (light text on dark). At ≥250 dpi the mask is
   smoothed by a 3×3 vote (≥5 of 9): single dots and ragged edges go away, and
   more letters match in the JBIG2 dictionary.
3. `background_image`: the page at the target resolution; pixels under the ink
   are filled by a normalized convolution of their neighbours (no halos).
4. `_background_search`: 100 dpi → (on success) 50 dpi, on failure 200 dpi. The
   JPEG quality is bisected on the SSIM of a "preview" (background + mask
   composite at 200 dpi) with a margin `PREVIEW_MARGIN`. JPEG 2000 is tried when
   `opj_compress` exists, from strong to weak compression (`JPX_RATES`, 2000:1 …
   50:1). On an almost flat background a JPEG is never smaller than about 4 KB
   per page; JPEG 2000 reaches about 1 KB.
5. `_foreground`: one colour per page. If the dark cores of the letters have
   several colours, a palette is used (`_palette_inks`): the colour is measured
   in cells of 1/40 inch (the dark half of the ink in a cell), inks differ
   mostly by chroma (lightness has weight 0.35, because thin strokes are
   lighter), up to 6 inks, 3×3 vote over the cells; each ink has its own mask
   with its own `rg`. If the cells are far from the palette (90th percentile >
   `PALETTE_FIT`) or there is just one ink, a 25 dpi colour image is painted
   through the mask.
6. Mask: JBIG2 with one shared dictionary per 64 masks (`-s`) when the total is
   smaller than Group 4/Flate; `--safe-text` uses generic JBIG2 without
   dictionaries. The symbol match threshold depends on the letter size
   (`stem_height`: the 90th percentile of the vertical ink runs, in px): ≥15 →
   0.80, ≥12 → 0.85, ≥10 → 0.90, otherwise generic and lossless
   (`JBIG2_THRESHOLDS`, `JBIG2_LOSSLESS`). No threshold is safe for glyphs below
   10 px: at 14 px (3.5 pt) even `-t 0.92` gave 333 swaps 9→8, and `-t 0.90`
   gave 410. Thresholds stay at 0.9 or below because jbig2enc 0.28 (Ubuntu
   24.04) rejects larger values and then the symbol mode does not work at all;
   its default is 0.85, so `-t` is always passed.
   The size is measured per band of lines between empty rows (`size_classes`),
   and the mask of every ink is split by threshold: a footnote under normal text
   gets its own mask with a strict threshold. Masks with one threshold are coded
   together, in chunks with a shared dictionary. 0.80 makes the mask two to
   three times smaller than 0.92 and matches `cjb2 -lossy` in size.
   The limits were measured on random digits with blur and noise at 300 dpi (a
   decoded glyph is classified against averaged templates): at 5 pt `-t 0.80`
   gave 62 swaps 6→8, at 4 pt `-t 0.85` gave 34 swaps 8↔6. Lines of digits at
   5 pt measure 13 px and were still confused at 0.80, hence the margin. With
   the chosen thresholds there were no swaps at any size, including 4–6 pt
   footnotes under 10 pt text; the split costs about 5 % of the size.
7. Page threshold: if `relaxed` (`--mrc-quality`) is given and the page is
   `text_like` (in every 128 px tile with ink, the composite "original as
   background + mask" keeps SSIM ≥ 0.85; text gives about 0.92, a coarsened
   photo about 0.6), the page is checked against `relaxed`, otherwise against
   the main threshold.
8. `build`: the layers are written into a Form XObject (`/PdfOptLayers`), the
   page is saved, `VisualGate` checks every page against its threshold, failing
   pages are restored (`restore`), and the document is saved again.

Pages get their own resource dictionaries (copies), so shared or inherited
`/Resources` of neighbouring pages are not affected.

## Lossy JPEG (`jpegopt.py`)

- `effective_dpis` parses the content stream (the `q`/`Q`/`cm` stack) and finds
  the dpi at which each image is shown.
- `reencode`: for the scales {1.0, 300, 225, 150 dpi} a bisection finds the
  lowest quality whose SSIM *at the check resolution (200 dpi)* is not below
  `threshold + SEARCH_MARGIN`; the smallest by bytes wins. The winner is rebuilt
  as baseline/progressive (and with mozjpeg, if present) with the same pixels.
- The rendering model follows poppler: shrinking averages by area (BOX),
  enlarging repeats pixels (NEAREST, up to 4×). Images below 200 dpi are judged
  enlarged, otherwise SSIM would not see their blockiness.
- `optimize._Run.jpeg_stage` checks pages with re-encoded images by real
  rendering; images of pages below the threshold are restored
  (`jpegopt.revert`), the others stay.
- Flate images with ≥1500 colours in the sample count as photos and become JPEG;
  graphics stay Flate.
- Forms created by MRC are skipped, so layers are not re-encoded twice.

## Parallelism

- Between files: `ProcessPoolExecutor`. The number of files at once is bounded
  by the cores, by the number of files and by free memory (about 700 MB per file
  with `-q`, 300 MB without); each file gets `cores // files` cores. `-j` sets
  the number of cores.
- Inside a file: MRC uses processes (`_bounded_map` keeps at most `2·jobs` pages
  in memory); images and rendering use threads (the heavy work is in `numpy`,
  `zlib` and `pdftoppm`, which do not hold the GIL); ECT uses threads.
- With `jobs ≥ 4` independent engines (MRC, Ghostscript, Hyper) run at the same
  time; each has its own `scratch` subfolder.

## Decisions and limits

- **A failure stays local.** An exception while layering one page leaves that
  page as it is and the other pages are layered; a failure of the whole MRC does
  not cancel the JPEG stage, and a failure of the JPEG stage does not cancel the
  MRC result that was already checked.
- **The colour space of a scan is kept.** If the image is ICCBased/CalRGB, the
  background (and the colour layer of RGB scans) gets the same space, and the
  mask colours are set in it with `/CS0 cs … sc` instead of DeviceRGB `rg`.
- **SSIM is blind to details finer than 200 dpi** and is averaged over a page
  that is mostly white. MRC adds the margin `PREVIEW_MARGIN` and checks the real
  rendering. Symbol JBIG2 can merge similar letters; the size-based threshold
  protects against this at the sizes that were tested, but SSIM does not see a
  swap, so use `--safe-text` for critical documents. Refinement (`jbig2 -r`, a
  "soft" matching that corrects swaps) is not used: the jbig2enc README says
  such files crash Acrobat. OCRmyPDF removed lossy JBIG2 altogether for the same
  reason.
- **The preview is a model of poppler's rendering.** On scaled pages (a normal
  300 dpi scan shown at 200 dpi) it is off by at most ±0.004 SSIM, which the
  `PREVIEW_MARGIN` covers. It overestimates noticeably (up to +0.025) only
  coloured text on a page shown 1:1: poppler restores the chroma of a JPEG in
  another way, and the difference sits in how the mask itself is drawn. Such a
  page does not pass the strict threshold and stays an image (it passes at the
  text threshold). Tried and rejected: poppler's own rendering as the preview
  reference (it mixes scaling models; the error on scans grows to −0.07) and
  retrying rejected pages with a target raised by the measured error (the
  background does not make up for that error).
- **The `/PdfOpt` marker** lives in the Catalog, not in Info, so the lossless
  metadata check is not disturbed. A file that gained nothing gets no marker and
  is analysed again next time.
- **Not processed:** signed PDFs, password-protected files (files that open
  without a password only with `--decrypt`; the result has no encryption),
  CMYK/Indexed JPEG, images with `/Decode` or colour-key masks in lossy mode,
  inline images, images inside forms when the dpi is determined.
- **Disk:** the work folder is created next to the file (for an atomic
  replace); with `-q` the reference renderings take about 1 MB per page.
- **Memory:** MRC pages are decoded whole (about 25 MB per A4 page at 300 dpi in
  numpy arrays, several copies); at most `2·jobs` pages are processed at once.
  Images above 512 MB unpacked (`MAX_SAMPLE_BYTES`) are skipped, and plain Flate
  is unpacked only to the size the image needs, so a decompression bomb cannot
  exhaust memory.
- **Open question: the `-q` scale is not smooth.** On scans, `-q 90`, `-q 85` and
  `-q 80` can give byte-identical files (the search has reached its floor: the
  lowest background dpi and JPEG quality), while `-q 96` is four times larger.
  Lowering the floors would make the scale smoother but needs measurements of the
  quality cost. See the TODO list in CHANGELOG.md.
