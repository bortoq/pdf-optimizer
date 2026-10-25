"""Standalone PDF optimizer: lossless recompression, lossy images and MRC."""

from .optimize import MIN_SSIM, optimize_pdf
from .pdfmodel import Skipped

__version__ = "0.2.0"
__all__ = ["MIN_SSIM", "Skipped", "__version__", "optimize_pdf"]
