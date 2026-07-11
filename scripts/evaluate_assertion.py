"""
Assertion Detection Evaluation Script

Evaluate assertion detection (isNegated, isFamily, isHistorical) on validation data.
Outputs: errors.csv and F1 report per assertion type.

Usage:
    python scripts/evaluate_assertion.py
    python scripts/evaluate_assertion.py --data data/validation/manual_validation_template.jsonl
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.assertion.rules import RuleBasedDetector


def load_validation_data(path: str) -> List[Dict[str, Any]]:
    """Load validation samples from JSONL."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def evaluate_sample(
    detector: RuleBasedDetector,
    sample: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Evaluate a single sample.

    Returns:
        - List of errors found (for errors.csv)
        - Dict of per-entity results with expected vs predicted
    """
    text = sample["text"]
    sample_id = sample.get("id", "unknown")
    errors = []
    results = {}

    for entity in sample.get("expected_entities", []):
        ent_text = entity["text"]
        ent_type = entity.get("type", "UNKNOWN")

        # Find entity position in text
        try:
            start = text.index(ent_text)
            end = start + len(ent_text)
        except ValueError:
            errors.append({
                "sample_id": sample_id,
                "text": text,
                "entity_text": ent_text,
                "error": "Entity text not found in text",
                "expected": entity.get("assertions", []),
                "predicted": [],
            })
            results[ent_text] = {
                "expected": entity.get("assertions", []),
                "predicted": [],
                "correct": False,
            }
            continue

        # Detect assertions
        result = detector.detect(text, start, end, entity_type=ent_type)

        # Build prediction
        predicted = []
        if result.status.is_negated:
            predicted.append("isNegated")
        if result.status.is_historical:
            predicted.append("isHistorical")
        if result.status.is_family:
            predicted.append("isFamily")

        expected = entity.get("assertions", [])

        # Check correctness
        correct = (set(expected) == set(predicted))

        results[ent_text] = {
            "start": start,
            "end": end,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
        }

        # Record errors
        if not correct:
            errors.append({
                "sample_id": sample_id,
                "text": text,
                "entity_text": ent_text,
                "entity_type": ent_type,
                "entity_start": start,
                "entity_end": end,
                "expected_assertions": "|".join(expected) if expected else "(none)",
                "predicted_assertions": "|".join(predicted) if predicted else "(none)",
                "is_neg_expected": "isNegated" in expected,
                "is_neg_predicted": "isNegated" in predicted,
                "is_hist_expected": "isHistorical" in expected,
                "is_hist_predicted": "isHistorical" in predicted,
                "is_fam_expected": "isFamily" in expected,
                "is_fam_predicted": "isFamily" in predicted,
            })

    return errors, results


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_assertion_metrics(
    all_results: List[Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, float]]:
    """Compute per-assertion metrics."""
    metrics = {}

    for assertion_type in ["isNegated", "isHistorical", "isFamily"]:
        tp = sum(
            1 for r in all_results
            for entity_result in r.values()
            if assertion_type in entity_result["expected"]
            and assertion_type in entity_result["predicted"]
        )
        fp = sum(
            1 for r in all_results
            for entity_result in r.values()
            if assertion_type not in entity_result["expected"]
            and assertion_type in entity_result["predicted"]
        )
        fn = sum(
            1 for r in all_results
            for entity_result in r.values()
            if assertion_type in entity_result["expected"]
            and assertion_type not in entity_result["predicted"]
        )

        metrics[assertion_type] = compute_metrics(tp, fp, fn)
        metrics[assertion_type]["tp"] = tp
        metrics[assertion_type]["fp"] = fp
        metrics[assertion_type]["fn"] = fn
        metrics[assertion_type]["support"] = tp + fn

    # Overall entity accuracy
    total = sum(len(r) for r in all_results)
    correct = sum(1 for r in all_results for v in r.values() if v["correct"])
    metrics["entity_accuracy"] = {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}

    return metrics


def write_errors_csv(errors: List[Dict[str, Any]], output_path: str):
    """Write errors to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id", "text", "entity_text", "entity_type",
            "entity_start", "entity_end",
            "expected_assertions", "predicted_assertions",
            "is_neg_expected", "is_neg_predicted",
            "is_hist_expected", "is_hist_predicted",
            "is_fam_expected", "is_fam_predicted",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for error in errors:
            row = {k: error.get(k, "") for k in fieldnames}
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate assertion detection")
    parser.add_argument(
        "--data",
        type=str,
        default="data/validation/manual_validation_template.jsonl",
        help="Validation data path"
    )
    parser.add_argument(
        "--output-errors",
        type=str,
        default="outputs/errors.csv",
        help="Output path for errors CSV"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="outputs/assertion_report.json",
        help="Output path for metrics report"
    )
    args = parser.parse_args()

    # Load data
    samples = load_validation_data(args.data)
    print(f"Loaded {len(samples)} validation samples")

    # Evaluate
    detector = RuleBasedDetector()
    all_errors = []
    all_results = []

    for sample in samples:
        errors, results = evaluate_sample(detector, sample)
        all_errors.extend(errors)
        all_results.append(results)

    # Compute metrics
    metrics = compute_assertion_metrics(all_results)

    # Print report
    print("\n" + "=" * 70)
    print("ASSERTION DETECTION EVALUATION REPORT")
    print("=" * 70)

    print(f"\nDataset: {args.data}")
    print(f"Total samples: {len(samples)}")

    total_entities = sum(len(r) for r in all_results)
    total_errors = len(all_errors)
    correct_entities = sum(1 for r in all_results for v in r.values() if v["correct"])

    print(f"Total entities evaluated: {total_entities}")
    print(f"Correct: {correct_entities}")
    print(f"Errors: {total_errors}")
    print(f"Entity Accuracy: {correct_entities/total_entities*100:.1f}%")

    print("\n" + "-" * 70)
    print(f"{'Assertion Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)

    for assertion_type in ["isNegated", "isHistorical", "isFamily"]:
        m = metrics[assertion_type]
        support = m.get("support", 0)
        tp = m.get("tp", 0)
        fp = m.get("fp", 0)
        fn = m.get("fn", 0)
        print(
            f"{assertion_type:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {support:>10}"
        )
        print(f"  (TP={tp}, FP={fp}, FN={fn})")

    print("-" * 70)

    # Overall micro F1
    total_tp = sum(metrics[t]["tp"] for t in ["isNegated", "isHistorical", "isFamily"])
    total_fp = sum(metrics[t]["fp"] for t in ["isNegated", "isHistorical", "isFamily"])
    total_fn = sum(metrics[t]["fn"] for t in ["isNegated", "isHistorical", "isFamily"])
    overall = compute_metrics(total_tp, total_fp, total_fn)
    print(f"{'Overall (micro)':<20} {overall['precision']:>10.4f} {overall['recall']:>10.4f} {overall['f1']:>10.4f}")
    print("=" * 70)

    # Write outputs
    Path(args.output_errors).parent.mkdir(parents=True, exist_ok=True)
    write_errors_csv(all_errors, args.output_errors)
    print(f"\nErrors CSV: {args.output_errors} ({len(all_errors)} errors)")

    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.data,
        "total_samples": len(samples),
        "total_entities": total_entities,
        "correct_entities": correct_entities,
        "entity_accuracy": correct_entities / total_entities if total_entities > 0 else 0.0,
        "metrics": {
            t: {
                "precision": metrics[t]["precision"],
                "recall": metrics[t]["recall"],
                "f1": metrics[t]["f1"],
                "support": metrics[t].get("support", 0),
                "tp": metrics[t].get("tp", 0),
                "fp": metrics[t].get("fp", 0),
                "fn": metrics[t].get("fn", 0),
            }
            for t in ["isNegated", "isHistorical", "isFamily"]
        },
        "overall": {
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
    }
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report JSON: {args.output_report}")


if __name__ == "__main__":
    main()
