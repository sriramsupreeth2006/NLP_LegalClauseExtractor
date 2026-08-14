"""Segment contracts into clause-sized text spans using spaCy and structure heuristics."""

from __future__ import annotations

import re

import spacy
from spacy.language import Language


SECTION_PATTERN = re.compile(
    r"\n(?=\s*(?:"
    r"\d+(?:\.\d+)*[.)]?|"
    r"Section\s+\d+(?:\.\d+)*|"
    r"[A-Z][A-Za-z0-9 ,/&()\-]{4,}:"
    r")\s+)",
    re.I,
)


def split_segments(text: str, nlp: Language) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    if len(paragraphs) > 1:
        return _merge_short_segments(paragraphs)

    section_splits = [part.strip() for part in SECTION_PATTERN.split(normalized) if part.strip()]
    if len(section_splits) > 1:
        return _merge_short_segments(section_splits)

    doc = nlp(normalized[:1000000])
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if len(sentences) <= 1:
        return [normalized]

    grouped: list[str] = []
    buffer: list[str] = []

    for sentence in sentences:
        buffer.append(sentence)
        if len(buffer) >= 2 or len(" ".join(buffer)) > 220:
            grouped.append(" ".join(buffer))
            buffer = []

    if buffer:
        if grouped:
            grouped[-1] = f"{grouped[-1]} {' '.join(buffer)}"
        else:
            grouped.append(" ".join(buffer))

    return _merge_short_segments(grouped)


def _merge_short_segments(segments: list[str]) -> list[str]:
    merged: list[str] = []
    for segment in segments:
        if merged and len(segment) < 80:
            merged[-1] = f"{merged[-1]}\n\n{segment}"
        else:
            merged.append(segment)
    return merged
