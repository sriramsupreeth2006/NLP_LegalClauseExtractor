"""CLI demo for NLP legal clause extraction presentations."""

from __future__ import annotations

import json
import textwrap

from nlp.pipeline import ClauseExtractionPipeline

SAMPLE = """
This Master Services Agreement ("Agreement") is entered into by Provider and Client.

1. Term and Termination. Either party may terminate for convenience upon ninety (90) days' prior written notice.
Provider may terminate immediately upon failure to cure a material breach within fifteen (15) days.

2. Fees and Payment Terms. Client shall pay a monthly fee of $18,500. Payment is due within thirty (30) days of invoice.
Any undisputed amount not paid within thirty (30) days shall accrue interest at 1.5% per month.

3. Late Payment Penalties. Provider may suspend Services and Client shall pay a late fee equal to 5% of outstanding balance.

4. Limitation of Liability. Aggregate liability shall not exceed fees paid in twelve months.

5. Early Termination Fee. Client shall pay an early termination fee equal to 50% of remaining contract value.
"""


def main() -> None:
    print("Loading NLP models (spaCy + DistilBERT)...")
    pipeline = ClauseExtractionPipeline()
    pipeline.load()

    result = pipeline.extract_from_text(SAMPLE.strip())
    print("\n=== NLP metadata ===")
    print(json.dumps(result["nlp"], indent=2))

    print(f"\n=== Extracted {len(result['clauses'])} clauses ===")
    for index, clause in enumerate(result["clauses"], start=1):
        print("\n" + "-" * 72)
        print(f"{index}. [{clause['category'].upper()}] {clause['title']} ({clause['confidence']:.0%} confidence)")
        print(f"Section: {clause['section']}")
        print(f"Summary: {clause['summary']}")
        if clause.get("entities"):
            entities = ", ".join(f"{item['label']}={item['text']}" for item in clause["entities"])
            print(f"Entities: {entities}")
        excerpt = textwrap.shorten(clause["excerpt"].replace("\n", " "), width=120, placeholder="...")
        print(f"Excerpt: {excerpt}")


if __name__ == "__main__":
    main()
