"""Tests. The cache of files that gained nothing goes to a throwaway folder,
so runs neither read nor change the user's own."""

import atexit
import os
import shutil
import tempfile

_cache = tempfile.mkdtemp(prefix="pdfopt-test-cache-")
os.environ["XDG_CACHE_HOME"] = _cache
atexit.register(shutil.rmtree, _cache, ignore_errors=True)
