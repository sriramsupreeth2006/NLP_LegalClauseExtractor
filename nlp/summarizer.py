"""Generate clause summaries using spaCy NER and linguistic patterns."""

from __future__ import annotations

import re

from spacy.language import Language


def infer_title(segment: str, category: str) -> str:
    cleaned = " ".join(segment.split())
    section_match = re.match(
        r"^\s*(?:Section\s+\d+(?:\.\d+)*|\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)]?)\s*"
        r"([^:.]{4,80}?)(?:[.:]|\s{2,}|\s+-\s+|$)",
        cleaned,
        re.I,
    )
    raw_title = section_match.group(1) if section_match else re.split(r"[.?!:;]", cleaned, maxsplit=1)[0]
    words = re.sub(r"^[-\d\s.()]+", "", raw_title).split()
    if words:
        return " ".join(words[:5])

    fallback = {
        "termination": "Termination clause",
        "payment": "Payment terms",
        "penalties": "Penalty clause",
        "other": "Risk clause",
    }
    return fallback.get(category, "Clause")


def infer_section(segment: str, fallback_index: int) -> str:
    explicit = re.match(
        r"^\s*(Section\s+\d+(?:\.\d+)*|\d+(?:\.\d+)*[.)]?|§\s*\d+(?:\.\d+)*)\b",
        segment.strip(),
        re.I,
    )
    if explicit:
        label = explicit.group(1).replace("  ", " ").strip()
        if label.startswith("Section") or label.startswith("§"):
            return label
        return f"Section {label.rstrip('.)')}"
    return f"Section {fallback_index}"


def extract_entities(segment: str, nlp: Language) -> list[dict[str, str]]:
    doc = nlp(segment[:100000])
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if not ent.text.strip() or key in seen:
            continue
        seen.add(key)
        entities.append({"text": ent.text.strip(), "label": ent.label_})

    for match in re.finditer(r"\$\s*[\d,]+(?:\.\d+)?", segment):
        key = (match.group(0), "MONEY")
        if key not in seen:
            seen.add(key)
            entities.append({"text": match.group(0), "label": "MONEY"})

    for match in re.finditer(r"\b\d+(?:\.\d+)?\s*%", segment):
        key = (match.group(0), "PERCENT")
        if key not in seen:
            seen.add(key)
            entities.append({"text": match.group(0), "label": "PERCENT"})

    return entities[:8]


def summarize_segment(segment: str, category: str, nlp: Language) -> str:
    lower = segment.lower()
    entities = extract_entities(segment, nlp)
    money = next((item["text"] for item in entities if item["label"] == "MONEY"), "")
    percent = next((item["text"] for item in entities if item["label"] == "PERCENT"), "")
    notice_days = _find_days(segment, r"(?:notice|terminate)[^.]{0,80}?(\d+)\s*(?:\(\d+\))?\s*(?:days?|business days?)", nlp)
    cure_days = _find_days(segment, r"cure[^.]{0,80}?(\d+)\s*(?:\(\d+\))?\s*(?:days?|business days?)", nlp)
    due_days = _find_days(segment, r"(?:due|within)[^.]{0,40}?(\d+)\s*(?:\(\d+\))?\s*(?:days?|business days?)", nlp)

    if category == "termination":
        if "for convenience" in lower and notice_days:
            return f"NLP detected a convenience termination clause with a {notice_days}-day notice requirement."
        if "breach" in lower and cure_days:
            return f"Termination may follow an uncured material breach after a {cure_days}-day cure period."
        if notice_days:
            return f"The agreement can be terminated with {notice_days} days' notice."
        return "Transformer classification identified language governing contract termination."

    if category == "payment":
        parts = []
        if money:
            parts.append(f"spaCy extracted payment amount {money}")
        if due_days:
            parts.append(f"invoices are due within {due_days} days")
        if percent:
            parts.append(f"late balances may accrue {percent} interest")
        if parts:
            return "Payment terms: " + ", ".join(parts) + "."
        return "Transformer classification identified payment and billing obligations."

    if category == "penalties":
        parts = []
        if percent:
            parts.append(f"a {percent} penalty or fee may apply")
        if "suspend" in lower:
            parts.append("services may be suspended for non-payment")
        if "early termination fee" in lower or "termination fee" in lower:
            return "NLP identified an early termination fee if the contract ends before the agreed term."
        if parts:
            return "Penalty language: " + ", ".join(parts) + "."
        return "Transformer classification identified penalty or default consequences."

    if "liability" in lower:
        return "Risk allocation clause: limits or defines liability between the parties."
    if "indemn" in lower:
        return "Indemnification clause requiring one party to cover specified losses or claims."
    if "confidential" in lower:
        return "Confidentiality clause restricting use and disclosure of sensitive information."
    return "Transformer classification identified an important contractual obligation or risk term."


def _find_days(segment: str, pattern: str, nlp: Language) -> str:
    paren_match = re.search(
        r"(?:ninety|sixty|thirty|forty-five|fifteen|twenty|ten|\d+)\s*\(\s*(\d+)\s*\)\s*(?:days?|business days?)",
        segment,
        re.I,
    )
    if paren_match:
        return paren_match.group(1)

    match = re.search(pattern, segment, re.I)
    if match:
        return match.group(1)

    days_match = re.search(r"\b(\d{1,3})\s*(?:\(\d+\))?\s*(?:days?|business days?)\b", segment, re.I)
    if days_match:
        return days_match.group(1)

    return ""
