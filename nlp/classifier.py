"""Zero-shot transformer classification for legal clause categories."""

from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import pipeline


CATEGORY_LABELS = {
    "termination": "contract termination, renewal, notice period, or breach cure",
    "payment": "payment terms, fees, invoicing, billing schedule, or interest on invoices",
    "penalties": "penalties, late fees, liquidated damages, suspension, or default consequences",
    "other": "liability cap, indemnity, confidentiality, warranty, insurance, or compliance obligations",
}

KEYWORD_HINTS = {
    "termination": [
        "terminate",
        "termination",
        "renewal",
        "notice period",
        "for convenience",
        "material breach",
        "cure period",
    ],
    "payment": [
        "payment",
        "invoice",
        "monthly fee",
        "billing",
        "interest",
        "due within",
        "fees paid",
    ],
    "penalties": [
        "penalty",
        "late fee",
        "liquidated damages",
        "suspend services",
        "early termination fee",
        "default",
    ],
    "other": [
        "limitation of liability",
        "indemn",
        "confidential",
        "warranty",
        "governing law",
        "insurance",
    ],
}

MIN_CONFIDENCE = 0.25
MAX_SEGMENT_CHARS = 1200
PREAMBLE_PATTERN = re.compile(
    r"^(?:this|whereas|now,? therefore).{0,120}(?:agreement|contract|parties)\b",
    re.I,
)


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    label_scores: dict[str, float]


class ClauseClassifier:
    model_name = "typeform/distilbert-base-uncased-mnli"

    def __init__(self) -> None:
        self._classifier = None

    def load(self) -> None:
        if self._classifier is None:
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                truncation=True,
            )

    def classify(self, segment: str) -> ClassificationResult | None:
        if self._classifier is None:
            raise RuntimeError("Classifier is not loaded.")

        text = " ".join(segment.split())
        if len(text) < 40:
            return None

        if PREAMBLE_PATTERN.search(text) and not re.search(r"^\s*\d+[.)]", text):
            return None

        candidate_labels = list(CATEGORY_LABELS.values())
        label_keys = list(CATEGORY_LABELS.keys())
        output = self._classifier(text[:MAX_SEGMENT_CHARS], candidate_labels, multi_label=False)

        scores = {
            label_keys[candidate_labels.index(label)]: float(score)
            for label, score in zip(output["labels"], output["scores"])
        }
        scores = self._blend_with_keyword_hints(text, scores)

        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]

        if confidence < MIN_CONFIDENCE:
            return None

        return ClassificationResult(
            category=best_category,
            confidence=confidence,
            label_scores=scores,
        )

    def _blend_with_keyword_hints(self, text: str, scores: dict[str, float]) -> dict[str, float]:
        lower = text.lower()
        blended = dict(scores)

        if "early termination fee" in lower or "termination fee" in lower:
            blended["penalties"] = blended.get("penalties", 0.0) + 0.35
            blended["termination"] = max(0.0, blended.get("termination", 0.0) - 0.1)

        for category, keywords in KEYWORD_HINTS.items():
            hits = sum(1 for keyword in keywords if keyword in lower)
            if hits:
                blended[category] = min(0.99, blended.get(category, 0.0) + hits * 0.08)

        total = sum(blended.values()) or 1.0
        return {category: value / total for category, value in blended.items()}
