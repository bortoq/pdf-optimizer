# pdfopt with every external tool it can use (poppler, jbig2enc, OpenJPEG,
# Ghostscript, mozjpeg's cjpeg comes with the Python package).
#   docker build -t pdfopt .
#   docker run --rm -v "$PWD":/data pdfopt -q 96 book.pdf
FROM python:3.12-slim-trixie

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential poppler-utils ghostscript libopenjp2-tools jbig2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN pip install --no-cache-dir . \
 && apt-get purge -y --auto-remove build-essential \
 && rm -rf /src

WORKDIR /data
ENTRYPOINT ["pdfopt"]
CMD ["--help"]
