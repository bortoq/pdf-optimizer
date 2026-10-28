# Development

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
