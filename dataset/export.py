import csv
import json
from pathlib import Path
from typing import List, Dict

from .sentences import SENTENCES


def export_to_csv(output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "text", "category", "subcategory",
        "contains_medical_terms", "medical_terms",
        "intended_tone", "expected_speech_rate",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sentence in SENTENCES:
            row = sentence.copy()
            row["medical_terms"] = "|".join(sentence["medical_terms"])
            writer.writerow(row)
    print(f"Exported {len(SENTENCES)} sentences to {output_path}")


def export_to_json(output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(SENTENCES, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(SENTENCES)} sentences to {output_path}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "results"
    export_to_csv(base / "dataset.csv")
    export_to_json(base / "dataset.json")
