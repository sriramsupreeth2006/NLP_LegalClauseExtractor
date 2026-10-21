import csv
import sys
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from ingestdocument import LegalDocumentIngester


class RealLegalDatasetManager:
    def __init__(self, dataset_csv: str = "clauses_dataset.csv"):
        self.dataset_path = Path(dataset_csv)
        self.target_categories = [
            "Governing Law",
            "Termination",
            "Indemnification",
            "Limitation of Liability",
            "Confidentiality",
            "Non-Compete",
            "General/Boilerplate"
        ]

    def infer_category(self, heading: str, body: str) -> str:
        """Heuristic rule-based taxonomy mapping to label real legal clauses."""
        combined = f"{heading} {body}".lower()

        # 1. Governing Law & Dispute Resolution
        if any(term in combined for term in [
            "governing law", "jurisdiction", "place of execution", 
            "courts at", "laws of india", "arbitration and conciliation",
            "dispute resolution"
        ]):
            return "Governing Law"

        # 2. Termination & Default
        if any(term in combined for term in [
            "events of defaults", "termination", "cancel/withdraw", 
            "cancellation", "condition of default", "refund to the allottee"
        ]):
            return "Termination"

        # 3. Indemnification & Defect Liability
        if any(term in combined for term in [
            "indemnify", "indemnification", "hold harmless", "defect liability",
            "defective title", "compensate the allottee", "structural defect"
        ]):
            return "Indemnification"

        # 4. Limitation of Liability
        if any(term in combined for term in [
            "limitation of liability", "indirect damages", 
            "consequential damages", "liability cap", "shall not exceed"
        ]):
            return "Limitation of Liability"

        # 5. Confidentiality & Non-Disclosure
        if any(term in combined for term in [
            "confidential", "proprietary information", 
            "non-disclosure", "trade secrets"
        ]):
            return "Confidentiality"

        # 6. Non-Compete & Restraint of Trade
        if any(term in combined for term in [
            "non-compete", "solicit", "restraint of trade", "competing business"
        ]):
            return "Non-Compete"

        # 7. General / Operational Boilerplate
        return "General/Boilerplate"

    def harvest_from_contracts(self, contract_files: List[str | Path]):
        """Ingests contracts, sanitizes PII/entities, labels clauses, and saves to CSV."""
        ingester = LegalDocumentIngester()
        harvested_records: List[Tuple[str, str, str]] = []

        # Track existing entries to prevent duplicate rows on re-runs
        existing_signatures = set()
        if self.dataset_path.exists():
            with open(self.dataset_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Signature based on the first 80 characters of clause text
                    existing_signatures.add(row["clause_text"][:80].strip())

        for file_path in contract_files:
            path = Path(file_path)
            if not path.exists():
                print(f"[!] Warning: Skipping missing file: {path}")
                continue

            print(f"[*] Ingesting and anonymizing: {path.name}")
            cleaned_text = ingester.extract_from_file(path)
            sections = ingester.segment_clauses(cleaned_text)

            new_clauses_count = 0
            for sec in sections:
                text_signature = sec.body[:80].strip()
                if text_signature not in existing_signatures:
                    label = self.infer_category(sec.heading, sec.body)
                    harvested_records.append((sec.body, label, sec.heading))
                    existing_signatures.add(text_signature)
                    new_clauses_count += 1

            print(f"[+] Harvested {new_clauses_count} new clauses from {path.name}")

        if not harvested_records:
            print("[*] No new clauses were added (dataset is already up to date).")
            return

        # Write or append records to CSV
        file_exists = self.dataset_path.exists()
        with open(self.dataset_path, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["clause_text", "label", "source_heading"])
            for text, label, heading in harvested_records:
                writer.writerow([text, label, heading])

        print(f"[✓] Saved {len(harvested_records)} newly harvested clauses to {self.dataset_path.name}")

    def load_dataset(self) -> List[Tuple[str, str]]:
        """Reads all labeled clauses from clauses_dataset.csv."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"{self.dataset_path} does not exist. Please harvest contracts first.")

        dataset = []
        with open(self.dataset_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset.append((row["clause_text"], row["label"]))
        return dataset

    def split_stratified(self, dataset: List[Tuple[str, str]], test_ratio: float = 0.2, seed: int = 42):
        """Performs a train-test split while preserving class distribution."""
        X = [text for text, _ in dataset]
        y = [label for _, label in dataset]

        return train_test_split(
            X, y, test_size=test_ratio, random_state=seed, stratify=y
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[!] Error: No input contract files provided.")
        print("Usage: python dataset_loader.py <contract1.pdf> [contract2.pdf ...]")
        sys.exit(1)

    contracts = sys.argv[1:]
    manager = RealLegalDatasetManager()
    
    print("[*] Starting dataset extraction and anonymization...")
    manager.harvest_from_contracts(contracts)
    
    current_dataset = manager.load_dataset()
    print(f"\n[✓] Total clauses currently in {manager.dataset_path.name}: {len(current_dataset)}")