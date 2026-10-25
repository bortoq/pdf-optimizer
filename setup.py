"""Builds the bundled ECT Deflate extension (everything else is in pyproject.toml)."""

from setuptools import Extension, setup

ECT = "pdfopt/_ect"

setup(ext_modules=[Extension(
    "pdfopt._ectdeflate",
    sources=[f"{ECT}/ectmodule.cpp", f"{ECT}/LzFind.c",
             f"{ECT}/zopfli/blocksplitter.c", f"{ECT}/zopfli/lz77.c",
             f"{ECT}/zopfli/squeeze.c", f"{ECT}/zopfli/util.c",
             f"{ECT}/zopfli/deflate.cpp", f"{ECT}/zopfli/katajainen.cpp"],
    include_dirs=[ECT],
    extra_compile_args=["-O2"],
    extra_link_args=["-pthread"],
    libraries=["stdc++", "m"],
)])
