"""
Step 2: Dataset Loader & Ground-Truth Aligner
Handles dataset creation, validation, loading, and stratified train/test splitting.
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class LabeledClause:
    """Represents an individual ground-truth labeled clause."""
    text: str
    label: str
    source_id: Optional[str] = None


class LegalDatasetManager:
    """Manages legal clause ground-truth datasets for supervised training."""

    # Canonical baseline dataset for the 7 target categories
    DEFAULT_CORPUS: List[Tuple[str, str]] = [
        # Governing Law
        ("This Agreement shall be governed by, and construed in accordance with, the laws of the State of Delaware, without regard to its conflict of laws principles.", "Governing Law"),
        ("The validity, interpretation, and performance of this Agreement shall be controlled by and construed under the laws of the State of New York.", "Governing Law"),
        ("This contract is made under and shall be governed by and construed in accordance with the laws of England and Wales.", "Governing Law"),
        ("Any disputes arising out of or related to this Agreement shall be resolved exclusively in the state and federal courts located in San Francisco, California.", "Governing Law"),
        ("This Agreement shall in all respects be interpreted, construed, and governed by the substantive laws of the Commonwealth of Massachusetts.", "Governing Law"),
        ("The parties agree that any legal action arising from this agreement will be subject to the exclusive jurisdiction of the courts of Texas.", "Governing Law"),

        # Termination
        ("Either party may terminate this Agreement without cause upon providing thirty (30) days prior written notice to the other party.", "Termination"),
        ("In the event of a material breach by either party, the non-breaching party may terminate this Agreement immediately if such breach remains uncured for 15 days.", "Termination"),
        ("Company reserves the right to terminate this contract immediately upon the occurrence of any bankruptcy or insolvency proceeding involving Provider.", "Termination"),
        ("This Agreement will automatically terminate at the expiration of the Initial Term unless renewed in writing by mutual agreement of the parties.", "Termination"),
        ("Either party may terminate this Agreement with immediate effect by written notice if the other party ceases to carry on its business.", "Termination"),
        ("Upon termination of this Agreement for any reason, all licenses granted hereunder shall immediately terminate.", "Termination"),

        # Indemnification
        ("Provider shall defend, indemnify, and hold harmless Company and its officers, directors, and employees against any third-party claims, liabilities, and damages.", "Indemnification"),
        ("Each party agrees to indemnify and hold harmless the other party from any losses or claims arising out of its gross negligence or willful misconduct.", "Indemnification"),
        ("Customer shall indemnify and defend the Vendor against any claims arising from Customer's unauthorized use of the software services.", "Indemnification"),
        ("The Contractor agrees to assume full responsibility and provide total indemnification for any property damage caused by its personnel during performance of the work.", "Indemnification"),
        ("Licensee shall indemnify, defend, and hold Licensor harmless from all liabilities resulting from Licensee's distribution of modified software.", "Indemnification"),

        # Limitation of Liability
        ("In no event shall either party's aggregate liability arising out of or related to this Agreement exceed the total fees paid by Customer in the preceding twelve (12) months.", "Limitation of Liability"),
        ("Neither party shall be liable to the other for any indirect, incidental, special, punitive, or consequential damages, including loss of profits or revenue.", "Limitation of Liability"),
        ("To the maximum extent permitted by applicable law, the total liability of Company for all claims under this Agreement shall not exceed $100,000.", "Limitation of Liability"),
        ("In no event will Licensor be liable for any lost data, business interruption, or indirect damages even if advised of the possibility thereof.", "Limitation of Liability"),
        ("The total cumulative liability of either party arising out of or in connection with this agreement will be strictly capped at the contract value.", "Limitation of Liability"),

        # Confidentiality
        ("Confidential Information means all non-public, proprietary technical, financial, or business information disclosed by one party to the other, whether verbally or in writing.", "Confidentiality"),
        ("The receiving party agrees to hold the disclosing party's Confidential Information in strict confidence and not disclose it to any third party for a period of five (5) years.", "Confidentiality"),
        ("Each party shall use the same degree of care to protect the confidentiality of the other party's information as it uses to protect its own confidential materials.", "Confidentiality"),
        ("All proprietary designs, source code, customer records, and trade secrets shared hereunder shall be treated as strictly Confidential Information.", "Confidentiality"),
        ("Recipient shall not use Discloser's proprietary information for any purpose other than performing its obligations under this Agreement.", "Confidentiality"),

        # Non-Compete
        ("During the term of this Agreement and for twelve (12) months thereafter, Employee shall not directly or indirectly engage in any business that competes with Company.", "Non-Compete"),
        ("Recipient shall not solicit or hire any active employee or independent contractor of the Discloser during the term and for one year following termination.", "Non-Compete"),
        ("The Consultant agrees not to provide similar consulting services to direct competitors operating within the geographic territory of North America.", "Non-Compete"),
        ("Contractor agrees not to solicit any customers or clients of the Company for a period of twenty-four (24) months after the termination of services.", "Non-Compete"),
        ("Executive shall not own, manage, operate, control, or participate in the ownership of any competing business entity within the defined territory.", "Non-Compete"),

        # General / Boilerplate
        ("This Agreement constitutes the entire agreement between the parties and supersedes all prior agreements, understandings, representations, and warranties.", "General/Boilerplate"),
        ("No amendment, modification, or waiver of any provision of this Agreement shall be effective unless in writing and signed by authorized representatives of both parties.", "General/Boilerplate"),
        ("If any provision of this Agreement is held to be invalid or unenforceable, the remaining provisions shall continue in full force and effect.", "General/Boilerplate"),
        ("Neither party may assign or transfer its rights under this Agreement without the prior written consent of the other party.", "General/Boilerplate"),
        ("All notices, requests, demands, and other communications under this Agreement shall be in writing and deemed delivered upon receipt via certified mail.", "General/Boilerplate"),
    ]

    def __init__(self, dataset_path: str = "clauses_dataset.csv"):
        self.dataset_path = Path(dataset_path)

    def initialize_csv_if_missing(self) -> None:
        """Creates the initial training CSV if one does not already exist."""
        if not self.dataset_path.exists():
            with open(self.dataset_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["text", "label"])
                for text, label in self.DEFAULT_CORPUS:
                    writer.writerow([text, label])
            print(f"[+] Initialized baseline ground-truth dataset at: {self.dataset_path}")

    def load_dataset(self) -> List[LabeledClause]:
        """Loads and validates labeled clause records from the CSV file."""
        self.initialize_csv_if_missing()

        clauses: List[LabeledClause] = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                text = row.get("text", "").strip()
                label = row.get("label", "").strip()
                if text and label:
                    clauses.append(LabeledClause(text=text, label=label, source_id=f"row_{idx}"))
                else:
                    print(f"[!] Warning: Skipping empty or malformed record at row {idx}")
        return clauses

    def get_class_distribution(self, clauses: List[LabeledClause]) -> Dict[str, int]:
        """Calculates frequency counts per category."""
        distribution: Dict[str, int] = {}
        for c in clauses:
            distribution[c.label] = distribution.get(c.label, 0) + 1
        return distribution

    def split_stratified(
        self,
        clauses: List[LabeledClause],
        test_ratio: float = 0.2,
        seed: int = 42
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Splits data into train and test sets while preserving class distribution ratios.
        Returns: (X_train, X_test, y_train, y_test)
        """
        by_class: Dict[str, List[str]] = {}
        for c in clauses:
            by_class.setdefault(c.label, []).append(c.text)

        train_x, train_y = [], []
        test_x, test_y = [], []

        rng = random.Random(seed)

        for label, texts in by_class.items():
            shuffled = list(texts)
            rng.shuffle(shuffled)
            
            # Ensure at least 1 sample goes to train and test when class size > 1
            n_samples = len(shuffled)
            if n_samples == 1:
                train_x.append(shuffled[0])
                train_y.append(label)
                continue

            test_count = max(1, int(n_samples * test_ratio))
            train_count = n_samples - test_count

            for t in shuffled[:train_count]:
                train_x.append(t)
                train_y.append(label)

            for t in shuffled[train_count:]:
                test_x.append(t)
                test_y.append(label)

        return train_x, test_x, train_y, test_y


if __name__ == "__main__":
    print("[*] Running Step 2: Dataset Loader & Ground-Truth Aligner Validation\n")
    manager = LegalDatasetManager()
    
    # Load and validate
    dataset = manager.load_dataset()
    print(f"[+] Total annotated records loaded: {len(dataset)}")

    # Display Class Balance
    dist = manager.get_class_distribution(dataset)
    print("\n[+] Class Distribution:")
    print("-" * 45)
    for label, count in sorted(dist.items()):
        print(f"  * {label:<25} : {count:>2} samples")
    print("-" * 45)

    # Perform Stratified Train/Test Split
    X_train, X_test, y_train, y_test = manager.split_stratified(dataset, test_ratio=0.2)
    print(f"\n[+] Train Partition: {len(X_train)} samples")
    print(f"[+] Test Partition : {len(X_test)} samples")
    print(f"[+] Partition Ratio: {len(X_train)/(len(dataset))*100:.1f}% Train / {len(X_test)/(len(dataset))*100:.1f}% Test")
    