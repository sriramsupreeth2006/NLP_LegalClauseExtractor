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
        # 1. Privacy & Redaction Rules
        self.redaction_rules = [
            (r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b", "[ID_REDACTED]"),
            (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "[TAX_ID_REDACTED]"),
            (r"\b[UL][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}\b", "[CIN_REDACTED]"),
            (r"(?:\+91[\-\s]?)?[6-9]\d{9}\b", "[PHONE_REDACTED]"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "[EMAIL_REDACTED]"),
            (r"(?i)scanned\s+by\s+camscanner", ""),
            (r"(?i)annexure\s*\[see\s*rule\s*\d+\]", ""),
            (
                r"(?i)\b[A-Z0-9\s,\.&]{3,50}\s+(?:CONSTRUCTIONS?|BUILDERS?|DEVELOPERS?|INFRASTRUCTURE|REALTORS?|PROPERTIES|ESTATES?|SOLUTIONS|SYSTEMS|CORP|INC|LIMITED|LTD|PVT\.?\s*LTD|LLP)\b",
                "[ORGANIZATION_REDACTED]"
            ),
            (
                r"(?i)(represented by\s+(?:its\s+)?(?:authorized\s+)?(?:signatory|partner|director|proprietor)?[\s,:]+)([A-Z][a-zA-Z\s\.]{2,35})(?=\s+(?:vide|having|hereinafter|aged|resident))",
                r"\1[REPRESENTATIVE_REDACTED]"
            ),
            (
                r"(?i)(son\s+of|daughter\s+of|wife\s+of|mr\.|ms\.|mrs\.|sri|shri)\s+([A-Z][a-zA-Z\s\.]{2,35})(?=\s+(?:aged|residing|hereinafter|,|\.))",
                r"\1 [INDIVIDUAL_REDACTED]"
            ),
        ]

        # 2. Section Heading Matcher: matches both numbered and labeled contract headings
        self.heading_pattern = re.compile(
            r"(?m)^\s*(?:\d{1,3}\s+)?(?:SECTION|CLAUSE|ARTICLE)?\s*(\d{1,2}(?:\.\d{1,2})*)\.?\s*[\n\t ]+([A-Z\s,/–\-]{3,65}?)(?::|\.|\n|$)"
        )

    def apply_redaction(self, text: str) -> str:
        cleaned = text
        for pattern, replacement in self.redaction_rules:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned

    def clean_text(self, raw_text: str) -> str:
        # Step A: Normalize line breaks
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # Step B: Remove leading OCR page numbers at the start of any line (e.g., "62 ", "63 [If...")
        text = re.sub(r"(?m)^\s*\d{1,3}\s+(?=[A-Za-z\[])", "", text)

        # Step C: Fix hyphenated line breaks
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # Step D: Ensure numbered clauses stand on their own line with double linebreaks
        # e.g., turn "some text 7. POSSESSION:" into "some text \n\n7. POSSESSION:"
        text = re.sub(
            r"([^\n])\s+(\b\d{1,2}\.\s+[A-Z\s,/–\-]{3,40}:)",
            r"\1\n\n\2",
            text
        )

        # Step E: Smooth out OCR single-word lines without destroying double newlines
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Step F: Collapse horizontal whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return self.apply_redaction(text.strip())

    def extract_from_file(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path.resolve()}")

        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            full_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            full_text = path.read_text(encoding="utf-8")

        return self.clean_text(full_text)

    def segment_clauses(self, full_text: str) -> List[ContractSection]:
        matches = list(self.heading_pattern.finditer(full_text))
        sections: List[ContractSection] = []

        if matches:
            for i, match in enumerate(matches):
                start_pos = match.start()
                end_pos = matches[i + 1].start() if (i + 1) < len(matches) else len(full_text)

                sec_num = match.group(1).strip()
                sec_title = " ".join(match.group(2).split())
                heading_full = f"{sec_num} {sec_title}"

                body_text = full_text[match.end():end_pos].strip()

                if len(body_text) >= 30:
                    sections.append(
                        ContractSection(
                            heading=heading_full,
                            body=body_text,
                            char_start=start_pos,
                            char_end=end_pos
                        )
                    )

        # Fallback if no numbered headings matched
        if not sections:
            blocks = [b.strip() for b in full_text.split("\n\n") if len(b.strip()) >= 50]
            for idx, block in enumerate(blocks, 1):
                first_line = block.split("\n")[0][:45]
                sections.append(
                    ContractSection(
                        heading=f"Clause Segment {idx}: {first_line}",
                        body=block,
                        char_start=0,
                        char_end=len(block)
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

        for idx, sec in enumerate(extracted_sections[:5], 1):
            print(f"--- [Section {idx}] Heading: {sec.heading} ---")
            print(f"Character Span: [{sec.char_start} : {sec.char_end}]")
            print(f"Body Preview: {sec.body[:150]}...\n")

    except Exception as e:
        print(f"[!] Processing failed: {e}")