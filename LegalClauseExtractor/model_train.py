import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from vectorizer_pipeline import LegalClauseVectorizer, LegalDomainFeatureExtractor


def train_clause_model(
    dataset_csv: str = "clauses_dataset.csv",
    vectorizer_path: str = "vectorizer.joblib",
    model_output: str = "clause_model.joblib",
    random_state: int = 42
):
    csv_file = Path(dataset_csv)
    vec_file = Path(vectorizer_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"[!] Dataset not found: {csv_file}")
    if not vec_file.exists():
        raise FileNotFoundError(f"[!] Vectorizer artifact not found: {vec_file}")

    print(f"[*] Loading training corpus from: {csv_file.name}")
    df = pd.read_csv(csv_file).dropna(subset=["clause_text", "label"])
    df["clause_text"] = df["clause_text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    print(f"[+] Total usable samples: {len(df)}")
    print("\n[+] Corpus category distribution:")
    class_counts = df["label"].value_counts()
    print(class_counts.to_string())
    print("-" * 65)

    # 1. Feature extraction using existing 17k-dimensional pipeline
    print(f"[*] Loading feature extraction pipeline from: {vec_file.name}")
    vec_pipeline = LegalClauseVectorizer.load(vec_file)
    X = vec_pipeline.transform(df["clause_text"])
    y = df["label"].values

    # 2. Determine K-Fold splits bounded by the smallest class
    min_class_count = int(class_counts.min())
    n_splits = max(2, min(min_class_count, 4))
    print(f"[*] Executing {n_splits}-Fold Stratified Cross-Validation on all {len(df)} clauses...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.empty(len(df), dtype=object)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_f, X_val_f = X[train_idx], X[val_idx]
        y_train_f = y[train_idx]

        clf_fold = LinearSVC(
            C=1.0,
            loss="squared_hinge",
            class_weight="balanced",
            random_state=random_state,
            max_iter=5000
        )
        clf_fold.fit(X_train_f, y_train_f)
        oof_predictions[val_idx] = clf_fold.predict(X_val_f)

    # 3. Print full out-of-fold metrics across all 118 clauses
    print("\n" + "=" * 65)
    print(f"   STRATIFIED OUT-OF-FOLD EVALUATION METRICS (TOTAL N={len(df)})")
    print("=" * 65)
    print(classification_report(y, oof_predictions, zero_division=0))
    print("=" * 65)

    # 4. Train final production classifier on 100% of available data
    print("\n[*] Training final production model on all 118 samples...")
    final_svc = LinearSVC(
        C=1.0,
        loss="squared_hinge",
        class_weight="balanced",
        random_state=random_state,
        max_iter=5000
    )
    final_svc.fit(X, y)

    # 5. Serialize final model artifact
    model_artifact = {
        "classifier": final_svc,
        "classes": list(final_svc.classes_),
        "vectorizer_path": str(vec_file)
    }
    joblib.dump(model_artifact, model_output)
    print(f"[✓] Final model successfully saved to: {model_output}\n")


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "clauses_dataset.csv"
    train_clause_model(dataset_csv=csv_arg)