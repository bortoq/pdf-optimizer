"""Layered PDF scan compression: JPEG background and a full-resolution stencil."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zlib

import numpy as np
from PIL import Image, ImageFilter, TiffImagePlugin
from pypdf.generic import (ArrayObject, BooleanObject, DictionaryObject,
                           EncodedStreamObject, NameObject, NumberObject)


PROFILES = ((0.50, 75), (0.60, 75), (0.75, 75))
PREVIEW_MARGIN = 0.012
JPEG_TIMEOUT = 600


def encode_jpeg_trials(bitmap: Image.Image, quality: int,
                       scratch: Path) -> list[bytes]:
    """Return JPEG candidates for ``bitmap``, smallest first.

    Pillow always supplies one candidate. When ``cjpeg`` is installed its
    baseline and progressive encodings are added so the caller can pick the
    smallest stream that still passes the quality gate.
    """
    candidates = []
    buffer = BytesIO()
    bitmap.save(buffer, "JPEG", quality=quality, optimize=True)
    candidates.append(buffer.getvalue())
    cjpeg = shutil.which("cjpeg")
    if cjpeg:
        suffix = ".ppm" if bitmap.mode == "RGB" else ".pgm"
        source = scratch / f"cjpeg-input{suffix}"
        target = scratch / "cjpeg-output.jpg"
        try:
            bitmap.save(source)
            for extra in ((), ("-progressive",)):
                target.unlink(missing_ok=True)
                completed = subprocess.run(
                    [cjpeg, "-quality", str(quality), "-optimize", *extra,
                     "-outfile", str(target), str(source)],
                    capture_output=True, text=True, check=False,
                    timeout=JPEG_TIMEOUT, stdin=subprocess.DEVNULL,
                )
                if completed.returncode == 0 and target.is_file():
                    candidates.append(target.read_bytes())
        except (OSError, subprocess.SubprocessError):
            pass
    return sorted(set(candidates), key=len)


def _image_stream(data: bytes, width: int, height: int, *, stencil: bool,
                  group4: bool = False):
    image = EncodedStreamObject()
    image._data = data
    image.update({
        NameObject("/Type"): NameObject("/XObject"),
        NameObject("/Subtype"): NameObject("/Image"),
        NameObject("/Width"): NumberObject(width),
        NameObject("/Height"): NumberObject(height),
        NameObject("/BitsPerComponent"): NumberObject(1 if stencil else 8),
        NameObject("/Filter"): NameObject("/CCITTFaxDecode" if group4 else
                                        "/FlateDecode" if stencil else "/DCTDecode"),
    })
    if stencil:
        image[NameObject("/ImageMask")] = BooleanObject(True)
        if group4:
            image[NameObject("/DecodeParms")] = DictionaryObject({
                NameObject("/K"): NumberObject(-1),
                NameObject("/Columns"): NumberObject(width),
                NameObject("/Rows"): NumberObject(height),
                NameObject("/BlackIs1"): BooleanObject(True),
            })
    else:
        image[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
    return image


def _encode_stencil(stencil: Image.Image) -> tuple[bytes, bool]:
    flate = zlib.compress(stencil.tobytes(), 9)
    try:
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[278] = stencil.height  # one strip, suitable for a PDF CCITT stream
        output = BytesIO()
        stencil.save(output, "TIFF", compression="group4", tiffinfo=tags)
        output.seek(0)
        tiff = Image.open(output)
        offsets, counts = tiff.tag_v2[273], tiff.tag_v2[279]
        if len(offsets) == len(counts) == 1:
            group4 = output.getvalue()[offsets[0]:offsets[0] + counts[0]]
            if len(group4) < len(flate):
                return group4, True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return flate, False


def _eligible(image) -> bool:
    filters = image.get("/Filter")
    return (filters == "/DCTDecode" or list(filters or []) ==
            ["/FlateDecode", "/DCTDecode"]) and (
            image.get("/ColorSpace") == "/DeviceRGB" and
            image.get("/BitsPerComponent") == 8 and
            image.get("/Width", 0) >= 500 and image.get("/Height", 0) >= 500 and
            not any(key in image for key in ("/Mask", "/SMask", "/Decode", "/DecodeParms")))


def _layers(source_bytes: bytes, stored_size: int,
            page_size: tuple[float, float], scratch: Path,
            compare, minimum: float):
    try:
        bitmap = Image.open(BytesIO(source_bytes)).convert("RGB")
    except (OSError, ValueError):
        return None
    width, height = bitmap.size
    grayscale = bitmap.convert("L")
    gray = np.asarray(grayscale, dtype=np.int16)
    blurred = np.asarray(grayscale.filter(ImageFilter.GaussianBlur(8)), dtype=np.int16)
    selected = (gray < 120) & (gray + 25 < blurred)
    fraction = float(selected.mean())
    if not 0.003 <= fraction <= 0.22:
        return None
    rgb = np.asarray(bitmap)
    color = tuple(int(v) for v in np.median(rgb[selected], axis=0))
    stencil = Image.fromarray((~selected * 255).astype(np.uint8), "L").convert("1")
    mask_bytes, group4 = _encode_stencil(stencil)
    selection = Image.fromarray((selected * 255).astype(np.uint8), "L")

    page_width = max(page_size[0], 1.0)
    page_height = max(page_size[1], 1.0)
    source_dpi = max(width * 72 / page_width, height * 72 / page_height)
    preview_scale = min(1.0, 200 / source_dpi)
    preview_size = (max(1, round(width * preview_scale)),
                    max(1, round(height * preview_scale)))
    reference = scratch / "mrc-reference.png"
    trial = scratch / "mrc-trial.png"
    bitmap.resize(preview_size, Image.Resampling.LANCZOS).save(reference)

    for scale, quality in PROFILES:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        background = bitmap.resize(size, Image.Resampling.LANCZOS)
        small_mask = selection.resize(size, Image.Resampling.BOX)
        small_mask = small_mask.filter(ImageFilter.MaxFilter(7))
        background.paste(background.filter(ImageFilter.MaxFilter(11)), mask=small_mask)
        jpeg = encode_jpeg_trials(background, quality, scratch)[0]
        if len(jpeg) + len(mask_bytes) + 600 >= stored_size:
            continue
        foreground = Image.open(BytesIO(jpeg)).resize(bitmap.size, Image.Resampling.BICUBIC)
        foreground.paste(color, mask=selection.convert("1"))
        foreground.resize(preview_size, Image.Resampling.LANCZOS).save(trial)
        if compare(reference, trial) >= min(1.0, minimum + PREVIEW_MARGIN):
            return jpeg, size, mask_bytes, (width, height), color, group4
    return None


def _layer_job(job):
    number, source_bytes, stored_size, page_size, scratch, compare, minimum = job
    directory = scratch / f"mrc-layer-{number}"
    directory.mkdir()
    try:
        return _layers(source_bytes, stored_size, page_size,
                       directory, compare, minimum)
    finally:
        shutil.rmtree(directory)


def add_layers(writer, scratch: Path, compare, minimum: float):
    """Replace one-image scanned page XObjects with forms; return undo records."""
    records = []
    total = len(writer.pages)
    images = []
    for page in writer.pages:
        resources = page.get("/Resources")
        xobjects = resources.get_object().get("/XObject") if resources else None
        found = ([ref.get_object() for ref in xobjects.get_object().values()
                  if ref.get_object().get("/Subtype") == "/Image"]
                 if xobjects else [])
        images.append(found[0] if len(found) == 1 else None)
    references = Counter(
        (id(image.indirect_reference.pdf), image.indirect_reference.idnum)
        for image in images if image is not None and image.indirect_reference is not None
    )
    def apply(number, image, result):
        if result is None:
            return
        jpeg, bg_size, mask, mask_size, color, group4 = result
        before = dict(image)
        before_data = image._data
        background = writer._add_object(_image_stream(jpeg, *bg_size, stencil=False))
        stencil = writer._add_object(_image_stream(mask, *mask_size, stencil=True,
                                                   group4=group4))
        image.clear()
        image.update({
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/FormType"): NumberObject(1),
            NameObject("/BBox"): ArrayObject([NumberObject(0), NumberObject(0),
                                               NumberObject(1), NumberObject(1)]),
            NameObject("/Resources"): DictionaryObject({
                NameObject("/XObject"): DictionaryObject({
                    NameObject("/Bg"): background,
                    NameObject("/Ink"): stencil,
                }),
            }),
        })
        image._data = (f"q /Bg Do Q q {color[0]/255:.5f} {color[1]/255:.5f} "
                       f"{color[2]/255:.5f} rg /Ink Do Q").encode("ascii")
        records.append((number, image, before, before_data))

    workers = min(4, os.cpu_count() or 1) if total >= 16 else 1
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    batch = []
    try:
        for number, page in enumerate(writer.pages, start=1):
            if total >= 50 and number % 50 == 0:
                print(f"PDF MRC: подготовлено {number}/{total} страниц",
                      file=sys.stderr, flush=True)
            image = images[number - 1]
            if image is None:
                continue
            reference = image.indirect_reference
            if (reference is None or
                    references[(id(reference.pdf), reference.idnum)] != 1 or
                    not _eligible(image)):
                continue
            job = (number, image.get_data(), len(image._data),
                   (float(page.mediabox.width), float(page.mediabox.height)),
                   scratch, compare, minimum)
            batch.append((number, image, job))
            if len(batch) == workers:
                jobs = [item[2] for item in batch]
                results = pool.map(_layer_job, jobs) if pool else map(_layer_job, jobs)
                for item, result in zip(batch, results):
                    apply(item[0], item[1], result)
                batch.clear()
        if batch:
            jobs = [item[2] for item in batch]
            results = pool.map(_layer_job, jobs) if pool else map(_layer_job, jobs)
            for item, result in zip(batch, results):
                apply(item[0], item[1], result)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    return records


def restore(record) -> None:
    _, image, dictionary, data = record
    image.clear()
    image.update(dictionary)
    image._data = data
