#!/usr/bin/env python3
"""
ICD-10 Candidate Retrieval Evaluation Script

Runs evaluation on icd_linking_samples.jsonl and outputs:
  - Recall@K metrics
  - retrieval_errors.csv
  - icd_retrieval_report.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.linking.icd.evaluator import ICDRetrievalEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICD-10 Candidate Retrieval")
    parser.add_argument(
        "--data",
        default="data/synthetic/icd_linking_samples.jsonl",
        help="Path to evaluation data",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k for evaluation (default: 20)",
    )
    parser.add_argument(
        "--errors",
        default=None,
        help="Output path for errors CSV (default: outputs/retrieval_errors.csv)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output path for report JSON (default: outputs/icd_retrieval_report.json)",
    )
    args = parser.parse_args()

    evaluator = ICDRetrievalEvaluator(
        data_path=args.data,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )

    print(f"Loading data from: {args.data}")
    metrics = evaluator.evaluate()

    # Print report
    evaluator.print_report()

    # Save outputs
    errors_path = evaluator.save_errors_csv(args.errors)
    report_path = evaluator.save_report(args.report)

    print(f"\nErrors CSV: {errors_path}")
    print(f"Report JSON: {report_path}")

    # Exit with error if recall@20 < 0.95
    recall_20 = metrics["recall"].get("recall@20", 0)
    if recall_20 < 0.95:
        print(f"\nWARNING: Recall@20 = {recall_20:.4f} < 0.95")
        sys.exit(1)
    else:
        print(f"\nRecall@20 = {recall_20:.4f} >= 0.95 — PASS")


if __name__ == "__main__":
    main()
