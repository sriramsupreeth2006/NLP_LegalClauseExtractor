import sys
import re
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


class LegalDomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Explicit Feature Engineering for Legal Clauses:
    Extracts structural, modal, and linguistic density signals.
    """
    def __init__(self):
        # 1. Deontic obligation modals (commands / requirements)
        self.deontic_modals = re.compile(
            r"\b(shall|must|agrees to|hereby|undertakes|warrants)\b", re.IGNORECASE
        )
        # 2. Defect / Liability / Indemnity signals
        self.liability_terms = re.compile(
            r"\b(liability|indemnify|indemnification|hold harmless|defective|defect|compensate|damages|loss)\b",
            re.IGNORECASE
        )
        # 3. Termination / Cancellation signals
        self.termination_terms = re.compile(
            r"\b(terminate|termination|default|cancel|cancellation|refund|rescind|breach)\b",
            re.IGNORECASE
        )
        # 4. Jurisdiction & Statutory Dispute signals
        self.jurisdiction_terms = re.compile(
            r"\b(court|courts|arbitration|governing law|jurisdiction|dispute|tribunal|laws of)\b",
            re.IGNORECASE
        )
        # 5. Financial / Payment signals
        self.payment_terms = re.compile(
            r"\b(rupees|rs|inr|percent|percentage|interest|installment|dues|payable|advance)\b",
            re.IGNORECASE
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            t = str(text)
            tokens = re.findall(r"\w+", t)
            num_tokens = max(len(tokens), 1)

            # A. Structural / Surface Features
            char_length = len(t)
            avg_word_length = char_length / num_tokens
            caps_ratio = sum(1 for w in tokens if w.isupper()) / num_tokens

            # B. Modal Verb & Obligation Density (per 100 words)
            modal_count = len(self.deontic_modals.findall(t))
            modal_density = (modal_count / num_tokens) * 100

            # C. Category Cue Densities (per 100 words)
            liability_density = (len(self.liability_terms.findall(t)) / num_tokens) * 100
            termination_density = (len(self.termination_terms.findall(t)) / num_tokens) * 100
            jurisdiction_density = (len(self.jurisdiction_terms.findall(t)) / num_tokens) * 100
            payment_density = (len(self.payment_terms.findall(t)) / num_tokens) * 100

            features.append([
                np.log1p(char_length),
                avg_word_length,
                caps_ratio,
                modal_density,
                liability_density,
                termination_density,
                jurisdiction_density,
                payment_density
            ])

        return np.array(features)


class LegalClauseVectorizer:
    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), sublinear_tf: bool = True):
        # FeatureUnion combines the sparse n-gram matrix and dense domain signals
        self.pipeline = FeatureUnion([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    stop_words="english",
                    sublinear_tf=sublinear_tf,
                    min_df=1,
                    max_df=0.95,
                    norm="l2",
                    strip_accents="unicode"
                )
            ),
            (
                "domain_features",
                LegalDomainFeatureExtractor()
            )
        ])
        self.classes_: List[str] = []

    def fit_transform(self, texts: pd.Series, labels: pd.Series):
        print(f"[*] Extracting hybrid features (TF-IDF + Domain Signals) on {len(texts)} samples...")
        X_feats = self.pipeline.fit_transform(texts)
        self.classes_ = sorted(labels.unique())
        print(f"[✓] Total feature dimensions extracted: {X_feats.shape[1]}")
        return X_feats

    def transform(self, texts: pd.Series | List[str]):
        return self.pipeline.transform(texts)

    def save(self, output_path: str | Path = "vectorizer.joblib"):
        path = Path(output_path)
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes_}, path)
        print(f"[✓] Engineered feature pipeline saved to: {path.name}")

    @classmethod
    def load(cls, artifact_path: str | Path = "vectorizer.joblib") -> "LegalClauseVectorizer":
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"[!] Artifact not found: {path}")
        data = joblib.load(path)
        instance = cls()
        instance.pipeline = data["pipeline"]
        instance.classes_ = data["classes"]
        return instance


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "clauses_dataset.csv"
    if not Path(csv_file).exists():
        print(f"[!] {csv_file} not found. Run dataset_loader.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_file).dropna(subset=["clause_text", "label"])
    df["clause_text"] = df["clause_text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    vectorizer = LegalClauseVectorizer()
    X = vectorizer.fit_transform(df["clause_text"], df["label"])
    vectorizer.save("vectorizer.joblib")