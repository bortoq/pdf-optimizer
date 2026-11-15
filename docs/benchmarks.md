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

## Books and papers

A set of 35 born-digital PDFs (135.9 MB: programming books, papers from
pdfTeX and dvips, office documents, one signed form), lossless, before and
after Type 1→CFF repacking, palettes and ECT for object streams:

| | Saved |
| --- | ---: |
| before | 34.2 MB (25.2 %) |
| after | **36.7 MB (27.0 %)** |

Papers with Type 1 fonts gain the most (per file, before → after): −6.8 →
**−72.2 %**, −2.6 → **−57.7 %**, −2.7 → **−56.7 %**, −23.7 → **−42.7 %**, a
487-page book −34.5 → **−44.3 %**. Object streams packed with ECT give
0.2–3 % on large books. All pages of the repacked fonts rendered pixel for
pixel as before (poppler at 100 dpi, checked on every file; Ghostscript at 100
and 300 dpi on samples).

Lossy (`-q 90`), files with screenshots, diagrams and gray pictures stored
as Flate; lowest page SSIM unchanged:

| File | Before | After |
| --- | ---: | ---: |
| 448 pages, mostly gray Flate images (28.9 MB) | −22.6 % | **−51.7 %** |
| 528 pages, gray Flate images (12.8 MB) | −18.1 % | **−29.3 %** |
| 928 pages, gray Flate images (11.9 MB) | −38.1 % | **−47.0 %** |
| 133 pages, flat RGB images (6.0 MB) | −50.4 % | **−86.9 %** |
| 5 pages, flat RGB images (1.9 MB) | −43.2 % | **−79.9 %** |

Other documents, measured with 0.2.0 (KB, source → pdfopt; `pikepdf`/qpdf in
brackets):

| Document | Result |
| --- | --- |
| dvips paper | 146 → **124** (130) |
| pdfTeX paper | 776 → **570** (747) |
| Word document | 1595 → **1073** (1532) |
| CCITT scan | 553 → **396** (539) |

An independent audit measured a 6-page 300 dpi scan (7.8 MB) at 111 KB with
SSIM 0.962, against 655 KB / 0.880 for DjVu (`pdf2djvu --lossy`).
