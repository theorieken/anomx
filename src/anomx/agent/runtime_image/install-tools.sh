#!/bin/sh
set -eu
tools_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash ca-certificates curl git jq ripgrep build-essential wget fd-find p7zip-full \
    poppler-utils imagemagick pandoc ffmpeg libimage-exiftool-perl \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu ghostscript qpdf \
    libreoffice-writer libreoffice-calc libreoffice-impress \
    libcairo2 libpango-1.0-0 fonts-dejavu fonts-liberation
rm -rf /var/lib/apt/lists/*
ln -sf /usr/bin/fdfind /usr/local/bin/fd
python -m pip install --no-cache-dir -r "$tools_dir/requirements.txt"
npm install --global pnpm@10
for tool in node npm pnpm uv rg fd 7z pdftotext pdftoppm convert pandoc ffmpeg exiftool tesseract gs qpdf libreoffice duckdb csvcut yq wget; do
    command -v "$tool" >/dev/null
done
tesseract --list-langs | grep -qx deu
python - <<'PY'
from importlib import import_module
for module in ("numpy", "pandas", "sklearn", "scipy", "narwhals", "joblib", "tqdm", "httpx", "requests", "pypdf", "pdfplumber", "reportlab", "PIL", "openpyxl", "docx", "pptx", "matplotlib", "bs4", "lxml", "duckdb", "csvkit", "qrcode", "barcode", "cairosvg", "fontTools", "jinja2", "rich", "pyarrow", "tabulate"):
    import_module(module)
PY
