"""Builds the bundled ECT Deflate extension (everything else is in pyproject.toml).

A failed build is not an error for a user (streams are then packed with zlib),
but it must be one where wheels are made: PDFOPT_REQUIRE_ECT=1 says so.
"""

import os
import sys

from setuptools import Extension, setup

ECT = "pdfopt/_ect"
MSVC = sys.platform == "win32"

setup(ext_modules=[Extension(
    "pdfopt._ectdeflate",
    sources=[f"{ECT}/ectmodule.cpp", f"{ECT}/LzFind.c",
             f"{ECT}/zopfli/blocksplitter.c", f"{ECT}/zopfli/lz77.c",
             f"{ECT}/zopfli/squeeze.c", f"{ECT}/zopfli/util.c",
             f"{ECT}/zopfli/deflate.cpp", f"{ECT}/zopfli/katajainen.cpp"],
    include_dirs=[ECT],
    optional=os.environ.get("PDFOPT_REQUIRE_ECT") != "1",
    extra_compile_args=["/O2"] if MSVC else ["-O2"],
    extra_link_args=[] if MSVC else ["-pthread"],
    libraries=[] if MSVC else ["stdc++", "m"],
)])
