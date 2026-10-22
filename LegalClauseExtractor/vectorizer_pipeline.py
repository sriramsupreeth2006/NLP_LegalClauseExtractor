import sys
from pathlib import Path
from typing import Tuple, List, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class LegalClauseVectorizer:
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        sublinear_tf: bool = True
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df,
            norm="l2",
            strip_accents="unicode"
        )
        self.classes_: List[str] = []

    def fit_transform(self, texts: pd.Series, labels: pd.Series):
        """Fits vectorizer vocabulary and transforms raw clause texts into sparse TF-IDF matrices."""
        print(f"[*] Extracting vocabulary and fitting TF-IDF on {len(texts)} samples...")
        X_tfidf = self.vectorizer.fit_transform(texts)
        self.classes_ = sorted(labels.unique())
        print(f"[✓] Vocabulary size: {len(self.vectorizer.vocabulary_)} distinct n-grams.")
        return X_tfidf

    def transform(self, texts: pd.Series | List[str]):
        """Transforms unseen texts using the fitted feature space."""
        return self.vectorizer.transform(texts)

    def inspect_top_features_per_class(
        self, X_tfidf, labels: pd.Series, top_n: int = 8
    ) -> Dict[str, List[str]]:
        """Identifies top-weighted n-grams per legal class based on mean TF-IDF scores."""
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_terms_by_class = {}

        for category in self.classes_:
            category_indices = np.where(labels.values == category)[0]
            if len(category_indices) == 0:
                continue

            # Compute mean TF-IDF weight across all clauses in this category
            mean_tfidf = np.asarray(X_tfidf[category_indices].mean(axis=0)).flatten()
            top_indices = mean_tfidf.argsort()[::-1][:top_n]
            top_terms_by_class[category] = feature_names[top_indices].tolist()

        return top_terms_by_class

    def save(self, output_path: str | Path = "vectorizer.joblib"):
        """Serializes vectorizer instance and class labels."""
        path = Path(output_path)
        artifact = {
            "vectorizer": self.vectorizer,
            "classes": self.classes_
        }
        joblib.dump(artifact, path)
        print(f"[✓] Vectorization pipeline saved to: {path.name}")

    @classmethod
    def load(cls, artifact_path: str | Path = "vectorizer.joblib") -> "LegalClauseVectorizer":
        """Loads a pre-fitted vectorizer artifact."""
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"[!] Vectorizer artifact not found: {path}")

        data = joblib.load(path)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.classes_ = data["classes"]
        return instance


def run_pipeline(
    input_csv: str = "clauses_dataset.csv",
    output_artifact: str = "vectorizer.joblib"
):
    csv_path = Path(input_csv)
    if not csv_path.exists():
        print(f"[!] Error: {input_csv} does not exist.")
        print("[!] Run dataset_loader.py first to create the dataset.")
        sys.exit(1)

    # 1. Load dataset
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["clause_text", "label"])
    df["clause_text"] = df["clause_text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    texts = df["clause_text"]
    labels = df["label"]

    print(f"[*] Dataset loaded successfully ({len(df)} rows).")

    # 2. Fit and transform
    feature_pipeline = LegalClauseVectorizer(ngram_range=(1, 2), min_df=1)
    X_tfidf = feature_pipeline.fit_transform(texts, labels)

    # 3. Inspect learned salient legal phrases
    print("\n" + "=" * 60)
    print("        TOP PREDICTIVE N-GRAMS BY LEGAL CATEGORY")
    print("=" * 60)
    top_terms = feature_pipeline.inspect_top_features_per_class(X_tfidf, labels, top_n=6)
    for category, terms in top_terms.items():
        print(f"[{category}]")
        print("  -> " + ", ".join(f"'{term}'" for term in terms))
    print("=" * 60 + "\n")

    # 4. Save artifact
    feature_pipeline.save(output_artifact)


if __name__ == "__main__":
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "clauses_dataset.csv"
    run_pipeline(input_csv=dataset_file)