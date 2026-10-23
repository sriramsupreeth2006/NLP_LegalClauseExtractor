import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Check for modern scikit-learn FrozenEstimator vs standard calibration
try:
    from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    from sklearn.calibration import CalibratedClassifierCV
    HAS_FROZEN = False

from vectorizer_pipeline import LegalClauseVectorizer, LegalDomainFeatureExtractor


def train_clause_model(
    dataset_csv: str = "clauses_dataset.csv",
    vectorizer_path: str = "vectorizer.joblib",
    model_output: str = "clause_model.joblib",
    test_size: float = 0.2,
    random_state: int = 42
):
    csv_file = Path(dataset_csv)
    vec_file = Path(vectorizer_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"[!] Dataset not found: {csv_file}")
    if not vec_file.exists():
        raise FileNotFoundError(
            f"[!] Vectorizer artifact not found: {vec_file}. Run vectorizer_pipeline.py first."
        )

    # 1. Load dataset
    print(f"[*] Loading training corpus from: {csv_file.name}")
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["clause_text", "label"])
    df["clause_text"] = df["clause_text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    print(f"[+] Total samples available: {len(df)}")
    print("\n[+] Sample breakdown per category:")
    counts = df["label"].value_counts()
    print(counts.to_string())
    print("-" * 60)

    # 2. Split dataset
    can_stratify = counts.min() >= 2
    if can_stratify:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            df["clause_text"],
            df["label"],
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"]
        )
    else:
        print("[!] Note: At least one class has < 2 samples. Running unstratified split.")
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            df["clause_text"],
            df["label"],
            test_size=test_size,
            random_state=random_state
        )

    print(f"[*] Split sizes: Training = {len(X_train_raw)} | Validation = {len(X_val_raw)}")

    # 3. Transform data using the hybrid pipeline
    print(f"[*] Loading feature extraction pipeline from: {vec_file.name}")
    vec_pipeline = LegalClauseVectorizer.load(vec_file)
    X_train = vec_pipeline.transform(X_train_raw)
    X_val = vec_pipeline.transform(X_val_raw)

    # 4. Train balanced LinearSVC classifier
    print("[*] Training Linear Support Vector Classifier with balanced class weights...")
    base_svc = LinearSVC(
        C=1.0,
        loss="squared_hinge",
        class_weight="balanced",
        random_state=random_state,
        max_iter=4000
    )
    base_svc.fit(X_train, y_train)

    # 5. Probability Calibration without cv='prefit'
    print("[*] Setting up probability calibration...")
    if HAS_FROZEN:
        # Modern scikit-learn (1.4+)
        calibrated_clf = CalibratedClassifierCV(estimator=FrozenEstimator(base_svc))
        calibrated_clf.fit(X_train, y_train)
    else:
        # Fallback if FrozenEstimator is unavailable
        try:
            calibrated_clf = CalibratedClassifierCV(estimator=base_svc, cv="prefit")
            calibrated_clf.fit(X_train, y_train)
        except Exception:
            # If cross-validation calibration fails due to single-instance class, use base_svc directly
            calibrated_clf = base_svc

    # 6. Model evaluation on validation split
    y_pred = calibrated_clf.predict(X_val)

    print("\n" + "=" * 60)
    print("               MODEL VALIDATION METRICS")
    print("=" * 60)
    print(classification_report(y_val, y_pred, zero_division=0))
    print("=" * 60)

    # 7. Serialize trained model and metadata
    classes = list(getattr(calibrated_clf, "classes_", getattr(base_svc, "classes_", [])))
    model_artifact = {
        "classifier": calibrated_clf,
        "classes": classes,
        "vectorizer_path": str(vec_file)
    }
    joblib.dump(model_artifact, model_output)
    print(f"[✓] Classifier successfully saved to: {model_output}\n")


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "clauses_dataset.csv"
    train_clause_model(dataset_csv=csv_arg)