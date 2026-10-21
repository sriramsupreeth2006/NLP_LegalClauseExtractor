import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
from pypdf import PdfReader


@dataclass
class ContractSection:
    heading: str
    body: str
    char_start: int
    char_end: int


class LegalDocumentIngester:
    def __init__(self):
        # 1. Privacy & Redaction Patterns (Entity names, identification numbers, artifacts)
        self.redaction_rules = [
            # 12-digit Indian national identity numbers
            (r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b", "[ID_REDACTED]"),
            # Tax identification / PAN codes
            (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "[TAX_ID_REDACTED]"),
            # Corporate Identification Numbers (CIN)
            (r"\b[UL][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}\b", "[CIN_REDACTED]"),
            # Phone / Contact numbers
            (r"(?:\+91[\-\s]?)?[6-9]\d{9}\b", "[PHONE_REDACTED]"),
            # Email addresses
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "[EMAIL_REDACTED]"),
            # Scanner stamps, watermarks, and regulatory filing headers
            (r"(?i)scanned\s+by\s+camscanner", ""),
            (r"(?i)annexure\s*\[see\s*rule\s*\d+\]", ""),
            # Enterprise, developer, and builder entities
            (
                r"(?i)\b[A-Z0-9\s,\.&]{3,50}\s+(?:CONSTRUCTIONS?|BUILDERS?|DEVELOPERS?|INFRASTRUCTURE|REALTORS?|PROJECTS?|PROPERTIES|ESTATES?|SOLUTIONS|SYSTEMS|CORP|INC|LIMITED|LTD|PVT\.?\s*LTD|LLP)\b",
                "[ORGANIZATION_REDACTED]"
            ),
            # Named party representations (e.g., "represented by its partner John Doe")
            (
                r"(?i)(represented by\s+(?:its\s+)?(?:authorized\s+)?(?:signatory|partner|director|proprietor)?[\s,:]+)([A-Z][a-zA-Z\s\.]{2,35})(?=\s+(?:vide|having|hereinafter|aged|resident))",
                r"\1[REPRESENTATIVE_REDACTED]"
            ),
            # Individual party names preceded by relational legal designations
            (
                r"(?i)(son\s+of|daughter\s+of|wife\s+of|mr\.|ms\.|mrs\.|sri|shri)\s+([A-Z][a-zA-Z\s\.]{2,35})(?=\s+(?:aged|residing|hereinafter|,|\.))",
                r"\1 [INDIVIDUAL_REDACTED]"
            ),
        ]

        # 2. Heading boundary regex for standard commercial and property contracts
        # Handles patterns like: "1. TERMS:", "ARTICLE 1 APPOINTMENT", "7. POSSESSION", "9. EVENTS OF DEFAULTS"
        self.heading_pattern = re.compile(
            r"(?m)^(?:\s*(?:SECTION|CLAUSE|ARTICLE)?\s*(\d{1,2}(?:\.\d{1,2})*)\.?\s+([A-Z\s,/–\-]{3,65}?)(?::|\.|\n))"
        )

    def apply_redaction(self, text: str) -> str:
        """Sanitizes sensitive identifiers and entities from the document text."""
        cleaned = text
        for pattern, replacement in self.redaction_rules:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned

    def clean_text(self, raw_text: str) -> str:
        """Normalizes formatting, de-hyphenates wrapped lines, and redacts PII."""
        # De-hyphenate words broken across line wraps
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw_text)
        # Collapse excessive horizontal spaces and blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Apply privacy and entity redaction
        return self.apply_redaction(text.strip())

    def extract_from_file(self, file_path: str | Path) -> str:
        """Reads text content from either a searchable PDF or a plain .txt file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path.resolve()}")

        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            full_text = path.read_text(encoding="utf-8")

        return self.clean_text(full_text)

    def segment_clauses(self, full_text: str) -> List[ContractSection]:
        """Splits full legal text into distinct clause sections using heading regex."""
        matches = list(self.heading_pattern.finditer(full_text))
        sections: List[ContractSection] = []

        if not matches:
            # Fallback for unnumbered contracts: chunk by natural double line breaks
            paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 80]
            for idx, p in enumerate(paragraphs, 1):
                sections.append(
                    ContractSection(
                        heading=f"Clause Segment {idx}",
                        body=p,
                        char_start=0,
                        char_end=len(p)
                    )
                )
            return sections

        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if (i + 1) < len(matches) else len(full_text)

            section_number = match.group(1)
            section_title = match.group(2).strip()
            heading_full = f"{section_number} {section_title}"

            body_text = full_text[match.end():end_pos].strip()

            if len(body_text) >= 40:
                sections.append(
                    ContractSection(
                        heading=heading_full,
                        body=body_text,
                        char_start=start_pos,
                        char_end=end_pos
                    )
                )

        return sections


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[!] Error: No input file provided.")
        print("Usage: python ingestdocument.py <path_to_contract_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    ingester = LegalDocumentIngester()
    print(f"[*] Processing document: {input_file}")

    try:
        sanitized_text = ingester.extract_from_file(input_file)
        extracted_sections = ingester.segment_clauses(sanitized_text)

        print(f"[✓] Successfully segmented {len(extracted_sections)} sections.\n")

        for idx, sec in enumerate(extracted_sections[:4], 1):
            print(f"--- [Section {idx}] Heading: {sec.heading} ---")
            print(f"Character Span: [{sec.char_start} : {sec.char_end}]")
            print(f"Body Preview: {sec.body[:150]}...\n")

    except Exception as e:
        print(f"[!] Processing failed: {e}")