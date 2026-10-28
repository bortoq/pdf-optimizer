"""In-process SSIM compatible with FFmpeg's ``ssim`` filter.

FFmpeg splits each plane into 4x4 blocks and scores 8x8 windows at a stride of
4 pixels; the same layout is used here so that thresholds keep their meaning
while no ``ffmpeg`` process has to be started for every page.
"""

from __future__ import annotations

import numpy as np

_C1 = (0.01 * 255) ** 2 * 64
_C2 = (0.03 * 255) ** 2 * 64 * 63
_STRIP = 256


def _block_sums(plane: np.ndarray) -> np.ndarray:
    """Sums over 4x4 blocks, using strided adds (much faster than reshape)."""
    rows = plane[0::4] + plane[1::4] + plane[2::4] + plane[3::4]
    return rows[:, 0::4] + rows[:, 1::4] + rows[:, 2::4] + rows[:, 3::4]


def _window(blocks: np.ndarray) -> np.ndarray:
    return (blocks[:-1, :-1] + blocks[1:, :-1] +
            blocks[:-1, 1:] + blocks[1:, 1:]).astype(np.float64)


def _plane_ssim(first: np.ndarray, second: np.ndarray) -> tuple[float, int]:
    """Return (sum of window scores, window count) for one 2-D plane."""
    height = first.shape[0] // 4 * 4
    width = first.shape[1] // 4 * 4
    if height < 8 or width < 8:
        return (1.0, 1) if np.array_equal(first, second) else (0.0, 1)
    total = 0.0
    count = 0
    # Strips overlap by one block row so every 8-row window is scored once.
    start = 0
    while start + 8 <= height:
        stop = min(height, start + _STRIP)
        # int32 keeps every block sum exact (at most 16 * 255 * 255).
        a = first[start:stop, :width].astype(np.int32)
        b = second[start:stop, :width].astype(np.int32)
        sa, sb = _window(_block_sums(a)), _window(_block_sums(b))
        saa = _window(_block_sums(a * a))
        sbb = _window(_block_sums(b * b))
        sab = _window(_block_sums(a * b))
        var_a = saa * 64 - sa * sa
        var_b = sbb * 64 - sb * sb
        cov = sab * 64 - sa * sb
        score = ((2 * sa * sb + _C1) * (2 * cov + _C2) /
                 ((sa * sa + sb * sb + _C1) * (var_a + var_b + _C2)))
        total += float(score.sum())
        count += score.size
        if stop >= height:
            break
        start = stop - 4
    return total, count


def ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Mean SSIM of two uint8 images of equal shape (H, W) or (H, W, C)."""
    if first.shape != second.shape:
        raise ValueError("SSIM: image sizes differ")
    if first.ndim == 2:
        first, second = first[:, :, None], second[:, :, None]
    total = 0.0
    count = 0
    for channel in range(first.shape[2]):
        part, size = _plane_ssim(first[:, :, channel], second[:, :, channel])
        total += part
        count += size
    return total / count
