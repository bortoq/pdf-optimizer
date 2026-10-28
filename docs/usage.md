# Usage

```sh
pdfopt book.pdf                      # lossless
pdfopt -q 96 book.pdf                # lossy, page SSIM >= 0.96
pdfopt --dry-run -r /path/to/books   # estimate the saving for a tree
pdfopt -v *.pdf                      # verbose log
pdfopt --decrypt locked.pdf          # encrypted, but opens without a password
pdfopt -o small.pdf book.pdf         # write a new file, keep the source
pdfopt -o out/ -j 4 *.pdf            # into out/, at most 4 cores
python3 -m pdfopt --help
```

| Option | Meaning |
| --- | --- |
| `-q`, `--quality N` | minimum page SSIM `N/100` (1–100); turns on lossy mode |
| `--dry-run` | compute the result without replacing the source |
| `-r`, `--recursive` | descend into subfolders |
| `-v`, `--verbose` | per-file results; on stderr the size and rejection reason of each engine, timings, warnings about missing programs |
| `--decrypt` | decrypt files that open without a password and optimize them (the owner's restrictions, such as printing or copying, are dropped) |
| `-o`, `--output PATH` | leave sources alone: write the result to a file (one input) or a directory (several); unchanged and skipped files are copied as they are |
| `-j`, `--jobs N` | cores to use (default: all) |
| `--version` | print the version |
| `--force` | process a file even if it is already optimized |
| `--safe-text` | with `-q`, code text masks without JBIG2 symbol mode (larger, but similar letters can never be swapped) |
| `--mrc-quality N` | minimum SSIM of MRC text pages (default 6 below `-q`; 100 with `-q 100`) |

**Exit codes:** `0` success (including "no change" and skipped files), `1` at
least one file failed, `2` bad arguments, or nothing to do because every path
named was rejected (a symlink, not a `.pdf`).

Without `-v` a terminal shows one progress line with the share saved so far,
and a single total at the end. Signed PDFs and password-protected files are
**skipped** (`skipped: ...`), which is not an error; files encrypted without a
password are processed only with `--decrypt`. Already optimized files are
counted in the total. When several files run at once, their number is bounded
by free memory (about 700 MB per file with `-q`).
