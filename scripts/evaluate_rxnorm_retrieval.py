#!/usr/bin/env python3
"""
CLI script for RxNorm drug retrieval evaluation.

Usage:
    python scripts/evaluate_rxnorm_retrieval.py
    python scripts/evaluate_rxnorm_retrieval.py --data data/synthetic/rxnorm_linking_samples.jsonl --top-k 10
    python scripts/evaluate_rxnorm_retrieval.py --errors --report
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Evaluate RxNorm drug retrieval")
    parser.add_argument(
        "--data",
        default="data/synthetic/rxnorm_linking_samples.jsonl",
        help="Path to evaluation samples JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K for retrieval (default: 10)",
    )
    parser.add_argument(
        "--errors",
        action="store_true",
        help="Save error CSV",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed report",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking",
    )
    parser.add_argument(
        "--cross-encoder",
        action="store_true",
        help="Use cross-encoder in reranker",
    )
    args = parser.parse_args()

    from src.linking.rxnorm import DrugRetrievalEvaluator

    evaluator = DrugRetrievalEvaluator(
        data_path=args.data,
        output_dir=args.output_dir,
        top_k=args.top_k,
        use_reranker=not args.no_rerank,
        use_cross_encoder=args.cross_encoder,
    )

    print(f"Running evaluation on {args.data}...")
    print(f"Knowledge base: {len(evaluator.entries)} entries")

    metrics = evaluator.evaluate()
    evaluator.print_report(metrics)
    evaluator.save_report(metrics)

    if args.errors:
        evaluator.save_errors_csv(metrics)


if __name__ == "__main__":
    main()
