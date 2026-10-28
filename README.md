# pdfopt

Shrink PDF files and prove the result is good. pdfopt tries several encoders,
checks every candidate by rendering the pages, and replaces your file only if
a verified result is smaller.

- **Lossless (default)**: text and images stay as they are; pages whose images
  or fonts changed are compared pixel for pixel.
- **Lossy (`-q N`)**: recompresses photos and scans while the SSIM of every page
  stays at or above `N/100`. A 9-page scan of 15 MB becomes about 100 KB.

Everything runs locally; no file leaves your machine.

## Install

```sh
pip install pdfopt
```

Python ≥ 3.11 and a C/C++ compiler are needed. For `-q` also install
`poppler-utils` (and, for the best scan results, `jbig2enc` and OpenJPEG), or
use Docker: `docker build -t pdfopt . && docker run --rm -v "$PWD":/data pdfopt -q 90 book.pdf`.
More in [docs/install.md](docs/install.md).

## Use

```sh
pdfopt book.pdf                 # lossless, replaces the file if smaller
pdfopt -q 90 scan.pdf           # lossy, every page keeps SSIM >= 0.90
pdfopt -o small.pdf book.pdf    # keep the source, write a new file
pdfopt --dry-run -r ~/books     # estimate the saving for a folder
```

## How it compares

Two 9-page 300 dpi scans (15 MB grayscale, 13 MB colour); lowest page SSIM
against the source:

| Tool | Gray size | Colour size | Lowest SSIM | Time (gray / colour) |
| --- | ---: | ---: | ---: | ---: |
| **pdfopt `-q 90`** | **104 KB** | **112 KB** | 0.94 / 0.95 | 13 s / 26 s |
| Ghostscript `/ebook` | 1,719 KB | 11,200 KB | 0.94 / 0.94 | 0.6 s / 2.7 s |
| Ghostscript `/screen` | 685 KB | 1,936 KB | 0.86 / 0.86 | 0.5 s / 1.4 s |
| **pdfopt (lossless)** | 13,049 KB | 11,359 KB | 1.00 | 1.4 s / 1.9 s |
| qpdf (lossless) | 14,938 KB | 12,914 KB | 1.00 | 0.1 s / 0.1 s |

At the same quality pdfopt is 7 to 100 times smaller than Ghostscript; the price
is time. Details and more documents: [docs/benchmarks.md](docs/benchmarks.md).

## Documentation

- [Usage and options](docs/usage.md)
- [Installation, Docker, external programs](docs/install.md)
- [How it works and how results are verified](docs/how-it-works.md)
- [Benchmarks](docs/benchmarks.md)
- [Architecture](docs/architecture.md) · [Development](docs/development.md) · [Changelog](CHANGELOG.md)

## License

MIT, see [LICENSE](LICENSE). The bundled ECT and Zopfli (`pdfopt/_ect`) are under
the Apache License 2.0, see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
