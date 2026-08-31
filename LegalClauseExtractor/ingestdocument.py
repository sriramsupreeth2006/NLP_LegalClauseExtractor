"""
Step 1: Document Ingestion & Section Segmentation
Reads PDF/TXT contracts, cleans layout artifacts, and segments text into sections.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


@dataclass
class ContractSection:
    section_id: int
    heading: str
    text: str
    char_start: int
    char_end: int


class ContractIngester:
    # Matches numbered sections (e.g. "1. DEFINITIONS"), Articles/Clauses, or uppercase headings
    HEADING_PATTERN = re.compile(
        r"(?m)^(?:\s*(?:ARTICLE|SECTION|CLAUSE)\s+[0-9IVXLCDM]+[\.\:\-]?[^\n]*|"
        r"\s*\d+\.\s+[A-Z\s]{2,50}|"
        r"\s*[A-Z][A-Z\s]{3,50}(?:\:|\b))"
    )

    def load_document(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() == ".pdf":
            if PdfReader is None:
                raise ImportError("pypdf is required. Run: pip install pypdf")
            reader = PdfReader(str(path))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            raw_text = "\n\n".join(pages_text)
        elif path.suffix.lower() in [".txt", ".text"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        return self.normalize_text(raw_text)

    def normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
        text = re.sub(r"(\b\w+)-\n\s*(\w+\b)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def segment_sections(self, normalized_text: str) -> List[ContractSection]:
        matches = list(self.HEADING_PATTERN.finditer(normalized_text))
        sections: List[ContractSection] = []

        if not matches:
            paragraphs = [p.strip() for p in normalized_text.split("\n\n") if p.strip()]
            curr_pos = 0
            for idx, p in enumerate(paragraphs, start=1):
                start = normalized_text.find(p, curr_pos)
                end = start + len(p)
                sections.append(ContractSection(
                    section_id=idx,
                    heading="General Provision",
                    text=p,
                    char_start=start,
                    char_end=end
                ))
                curr_pos = end
            return sections

        if matches[0].start() > 0:
            preamble = normalized_text[:matches[0].start()].strip()
            if preamble:
                sections.append(ContractSection(
                    section_id=1,
                    heading="Preamble / Recitals",
                    text=preamble,
                    char_start=0,
                    char_end=matches[0].start()
                ))

        for i, match in enumerate(matches):
            raw_heading = match.group(0).strip().split("\n")[0].strip()

            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(normalized_text)
            
            section_full = normalized_text[start_pos:end_pos].strip()

            # Separate heading line from clause body
            lines = [l.strip() for l in section_full.split("\n") if l.strip()]
            if lines:
                heading = lines[0]
                body_text = "\n".join(lines[1:]).strip() if len(lines) > 1 else lines[0]
            else:
                heading = raw_heading
                body_text = section_full

            sections.append(ContractSection(
                section_id=len(sections) + 1,
                heading=heading,
                text=body_text,
                char_start=start_pos,
                char_end=end_pos
            ))

        return sections


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingestdocument.py <path_to_contract.pdf_or_txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    ingester = ContractIngester()
    
    print(f"[*] Ingesting and processing: {input_file}")
    text = ingester.load_document(input_file)
    sections = ingester.segment_sections(text)

    print(f"[+] Successfully parsed {len(sections)} sections.\n")
    print("=" * 70)
    for sec in sections:
        print(f"Section ID  : {sec.section_id}")
        print(f"Heading     : {sec.heading}")
        print(f"Span        : [{sec.char_start}:{sec.char_end}]")
        print(f"Text Preview: {sec.text[:120]}...")
        print("-" * 70)