"""End-to-end NLP pipeline for legal clause extraction."""

from __future__ import annotations

import spacy

from nlp.classifier import ClauseClassifier
from nlp.document_loader import extract_text_from_bytes, normalize_text
from nlp.segmenter import split_segments
from nlp.summarizer import extract_entities, infer_section, infer_title, summarize_segment


class ClauseExtractionPipeline:
    spacy_model = "en_core_web_sm"

    def __init__(self) -> None:
        self._nlp = None
        self._classifier = ClauseClassifier()
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        self._nlp = spacy.load(self.spacy_model)
        if "sentencizer" not in self._nlp.pipe_names and "parser" not in self._nlp.pipe_names:
            self._nlp.add_pipe("sentencizer")
        self._classifier.load()
        self._loaded = True

    def extract_from_text(self, text: str) -> dict:
        if not self._loaded:
            self.load()

        normalized = normalize_text(text)
        if not normalized:
            return {
                "text": "",
                "clauses": [],
                "nlp": self._metadata(segment_count=0),
            }

        segments = split_segments(normalized, self._nlp)
        clauses = []

        for segment in segments:
            result = self._classifier.classify(segment)
            if result is None:
                continue

            entities = extract_entities(segment, self._nlp)
            clauses.append(
                {
                    "category": result.category,
                    "confidence": round(result.confidence, 3),
                    "label_scores": {key: round(value, 3) for key, value in result.label_scores.items()},
                    "title": infer_title(segment, result.category),
                    "summary": summarize_segment(segment, result.category, self._nlp),
                    "section": infer_section(segment, len(clauses) + 1),
                    "excerpt": segment,
                    "entities": entities,
                }
            )

        return {
            "text": normalized,
            "clauses": clauses,
            "nlp": self._metadata(segment_count=len(segments)),
        }

    def extract_from_file(self, data: bytes, filename: str) -> dict:
        text = extract_text_from_bytes(data, filename)
        return self.extract_from_text(text)

    def _metadata(self, segment_count: int) -> dict:
        return {
            "segment_count": segment_count,
            "classifier_model": self._classifier.model_name,
            "spacy_model": self.spacy_model,
            "techniques": [
                "spaCy tokenization and sentence segmentation",
                "spaCy named entity recognition (NER)",
                "DistilBERT zero-shot natural language inference (NLI)",
            ],
        }
