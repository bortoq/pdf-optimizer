# Benchmarks

## Scans (lossy)

Two 9-page 300 dpi scans (grayscale 14.9 MB, colour 12.9 MB), one machine
(Ryzen 5 5650U, 6 cores / 12 threads). `pdfopt -q 90` against Ghostscript
presets; SSIM is the lowest page score at 150 dpi against the source.

| File | Tool | Size | Lowest page SSIM | Time |
| --- | --- | ---: | ---: | ---: |
| grayscale | pdfopt `-q 90` | **104 KB** | 0.940 | 13 s |
| | Ghostscript `/ebook` | 1,719 KB | 0.935 | 0.6 s |
| | Ghostscript `/screen` | 685 KB | 0.855 | 0.5 s |
| colour | pdfopt `-q 90` | **112 KB** | 0.953 | 26 s |
| | Ghostscript `/ebook` | 11,200 KB | 0.941 | 2.7 s |
| | Ghostscript `/screen` | 1,936 KB | 0.862 | 1.4 s |

## Lossless

Same scans, byte-exact pixels (SSIM 1.000):

| File | Source | qpdf (pikepdf) | pdfopt | Time |
| --- | ---: | ---: | ---: | ---: |
| grayscale | 14,939 KB | 14,938 KB | **13,049 KB** | 1.4 s |
| colour | 12,916 KB | 12,914 KB | **11,359 KB** | 1.9 s |

Other documents (KB, source → pdfopt; `pikepdf`/qpdf in brackets):

| Document | Result |
| --- | --- |
| dvips paper | 146 → **124** (130) |
| pdfTeX paper | 776 → **570** (747) |
| Word document | 1595 → **1073** (1532) |
| CCITT scan | 553 → **396** (539) |

An independent audit measured a 6-page 300 dpi scan (7.8 MB) at 111 KB with
SSIM 0.962, against 655 KB / 0.880 for DjVu (`pdf2djvu --lossy`).
