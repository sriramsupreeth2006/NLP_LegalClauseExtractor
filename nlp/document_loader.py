"""Load and normalize contract text from raw strings or uploaded files."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


def normalize_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\u0000", "").strip()


def extract_text_from_bytes(data: bytes, filename: str = "") -> str:
    extension = Path(filename or "").suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return normalize_text("\n".join(pages))

    if extension == ".docx":
        document = Document(BytesIO(data))
        return normalize_text("\n".join(paragraph.text for paragraph in document.paragraphs))

    if extension in {".html", ".htm"}:
        raw = data.decode("utf-8", errors="ignore")
        raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
        raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.I)
        raw = re.sub(r"<[^>]+>", " ", raw)
        return normalize_text(raw)

    return normalize_text(data.decode("utf-8", errors="ignore"))
