# Installation

```sh
pip install pdfopt          # from PyPI (builds from source for now)
pip install git+https://github.com/bortoq/pdf-optimizer.git   # latest from GitHub
pip install -e .            # from a checkout: pikepdf, pypdf, Pillow, numpy, fontTools, mozjpeg + builds ECT
pip install -e '.[dev]'     # + pytest, ruff
```

Python ≥ 3.11 and C/C++ compilers are needed (the ECT module is built on
install; if the build fails, installation still works and streams are packed
with zlib). Running straight from a source checkout: `python3 setup.py build_ext --inplace`.
External programs are listed [below](#external-programs).

## Docker

**Docker** (all programs included):

```sh
docker build -t pdfopt .
docker run --rm -v "$PWD":/data pdfopt -q 96 book.pdf
```

## External programs

| Program | Used for | Without it |
| --- | --- | --- |
| `pdftoppm` (poppler) | render checks, **required for `-q`** | lossless drops image transforms; `-q` unavailable |
| `jbig2` (jbig2enc) | JBIG2 generic and symbol dictionaries | Group 4/Flate |
| `opj_compress` (OpenJPEG) | JPEG 2000 backgrounds | JPEG only |
| `cjpeg` from mozjpeg | denser JPEG | Pillow/libjpeg |
| `gs` (Ghostscript) | Ghostscript profiles | skipped |
| `hpdf-worker` (in `PATH`) | Hyper Compress | skipped with a warning |
