#!/usr/bin/env python3
"""
Ablation study: compare retrieval vs retrieval+rule reranking.

Usage:
    python scripts/ablation_reranking.py
    python scripts/ablation_reranking.py --icd-only
    python scripts/ablation_reranker.py --rx-only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ─── ICD-10 Ablation ────────────────────────────────────────────────────────


def run_icd_ablation():
    """Run ablation on ICD-10 retrieval."""
    from src.linking.icd.schema import get_knowledge_base
    from src.linking.icd.hybrid_retriever import HybridRetriever, MergeConfig, CandidateResult
    from src.linking.rule_reranker import ICDRuleReranker

    entries = get_knowledge_base()
    retriever = HybridRetriever(
        entries=entries,
        merge_config=MergeConfig(method="rrf", rrf_k=60),
        top_k=20,
    )
    reranker = ICDRuleReranker(entries)

    # Load samples
    data_path = Path("data/synthetic/icd_linking_samples.jsonl")
    if not data_path.exists():
        print(f"[ICD Ablation] Data not found: {data_path}")
        return None

    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    results_before = []
    results_after = []

    for sample in samples:
        gold = sample.get("positive_code", "")
        mention = sample.get("mention", "")
        query = sample.get("query_text", mention)

        candidates = retriever.retrieve(query, mention=mention, top_k=20)
        retrieved_codes = [c.code for c in candidates]

        # Before: use retrieval rank
        rank_before = retrieved_codes.index(gold) + 1 if gold in retrieved_codes else None
        results_before.append(rank_before)

        # After: use reranked rank
        reranked = reranker.rerank(candidates, query, mention=mention, top_k=20)
        reranked_codes = [r.code for r in reranked]
        rank_after = reranked_codes.index(gold) + 1 if gold in reranked_codes else None
        results_after.append(rank_after)

    return compute_metrics(results_before, results_after, samples)


def compute_metrics(ranks_before, ranks_after, samples):
    """Compute and compare metrics."""
    n = len(ranks_before)
    assert n > 0, "No samples"

    def recall_at_k(ranks, k):
        return sum(1 for r in ranks if r is not None and r <= k) / n

    def mrr(ranks):
        valid = [1.0 / r for r in ranks if r is not None]
        return sum(valid) / n if valid else 0.0

    def top1_acc(ranks):
        return sum(1 for r in ranks if r == 1) / n

    def better(ranks_b, ranks_a):
        """Count cases where reranking improved, worsened, same."""
        better_count = 0
        worse_count = 0
        same_count = 0
        for b, a in zip(ranks_b, ranks_a):
            if b is None and a is not None:
                better_count += 1  # moved from miss to hit
            elif b is not None and a is None:
                worse_count += 1
            elif b is not None and a is not None:
                if a < b:
                    better_count += 1
                elif a > b:
                    worse_count += 1
                else:
                    same_count += 1
            else:
                same_count += 1
        return better_count, worse_count, same_count

    better_c, worse_c, same_c = better(ranks_before, ranks_after)

    metrics = {
        "n_samples": n,
        "before": {
            "recall@1": round(recall_at_k(ranks_before, 1), 4),
            "recall@3": round(recall_at_k(ranks_before, 3), 4),
            "recall@5": round(recall_at_k(ranks_before, 5), 4),
            "recall@10": round(recall_at_k(ranks_before, 10), 4),
            "mrr": round(mrr(ranks_before), 4),
            "top1_acc": round(top1_acc(ranks_before), 4),
        },
        "after": {
            "recall@1": round(recall_at_k(ranks_after, 1), 4),
            "recall@3": round(recall_at_k(ranks_after, 3), 4),
            "recall@5": round(recall_at_k(ranks_after, 5), 4),
            "recall@10": round(recall_at_k(ranks_after, 10), 4),
            "mrr": round(mrr(ranks_after), 4),
            "top1_acc": round(top1_acc(ranks_after), 4),
        },
        "delta": {
            "recall@1": round(recall_at_k(ranks_after, 1) - recall_at_k(ranks_before, 1), 4),
            "recall@3": round(recall_at_k(ranks_after, 3) - recall_at_k(ranks_before, 3), 4),
            "recall@5": round(recall_at_k(ranks_after, 5) - recall_at_k(ranks_before, 5), 4),
            "recall@10": round(recall_at_k(ranks_after, 10) - recall_at_k(ranks_before, 10), 4),
            "mrr": round(mrr(ranks_after) - mrr(ranks_before), 4),
            "top1_acc": round(top1_acc(ranks_after) - top1_acc(ranks_before), 4),
        },
        "cases": {
            "better": better_c,
            "worse": worse_c,
            "same": same_c,
        }
    }
    return metrics


def run_rxnorm_ablation():
    """Run ablation on RxNorm retrieval."""
    from src.linking.rxnorm.schema import get_knowledge_base
    from src.linking.rxnorm.hybrid_retriever import DrugHybridRetriever
    from src.linking.rxnorm.parser import DrugMentionParser
    from src.linking.rule_reranker import RxNormRuleReranker

    entries = get_knowledge_base()
    retriever = DrugHybridRetriever(entries=entries, top_k=20)
    reranker = RxNormRuleReranker(entries)
    parser = DrugMentionParser()

    # Load samples
    data_path = Path("data/synthetic/rxnorm_linking_samples.jsonl")
    if not data_path.exists():
        print(f"[RxNorm Ablation] Data not found: {data_path}")
        return None

    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    ranks_before = []
    ranks_after = []

    for sample in samples:
        gold = sample.get("positive_rxcui", "")
        mention = sample.get("mention", "")
        query = sample.get("query_text", mention)

        candidates = retriever.retrieve(query, mention=mention, top_k=20)
        retrieved = [c.rxcui for c in candidates]

        rank_b = retrieved.index(gold) + 1 if gold in retrieved else None
        ranks_before.append(rank_b)

        reranked = reranker.rerank(candidates, query, mention=mention, top_k=20)
        reranked_codes = [r.code for r in reranked]
        rank_a = reranked_codes.index(gold) + 1 if gold in reranked_codes else None
        ranks_after.append(rank_a)

    return compute_metrics(ranks_before, ranks_after, samples)


def print_report(name, metrics):
    """Print ablation report."""
    print(f"\n{'='*60}")
    print(f"  {name} — Ablation Study")
    print(f"{'='*60}")
    print(f"  Samples: {metrics['n_samples']}")
    print()
    print(f"  {'Metric':<12} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for key in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "top1_acc"]:
        b = metrics["before"][key]
        a = metrics["after"][key]
        d = metrics["delta"][key]
        delta_str = f"+{d:.4f}" if d > 0 else f"{d:.4f}"
        print(f"  {key:<12} {b:>8.4f} {a:>8.4f} {delta_str:>8}")
    print()
    c = metrics["cases"]
    print(f"  Cases: better={c['better']}, worse={c['worse']}, same={c['same']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Ablation study for reranking")
    parser.add_argument("--icd-only", action="store_true", help="Only run ICD-10 ablation")
    parser.add_argument("--rx-only", action="store_true", help="Only run RxNorm ablation")
    parser.add_argument("--output", default="outputs/ablation_reranking.json", help="Output path")
    args = parser.parse_args()

    all_metrics = {}

    if not args.rx_only:
        print("\nRunning ICD-10 ablation...")
        icd_metrics = run_icd_ablation()
        if icd_metrics:
            print_report("ICD-10", icd_metrics)
            all_metrics["icd"] = icd_metrics

    if not args.icd_only:
        print("Running RxNorm ablation...")
        rx_metrics = run_rxnorm_ablation()
        if rx_metrics:
            print_report("RxNorm", rx_metrics)
            all_metrics["rxnorm"] = rx_metrics

    if all_metrics:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nAblation report saved to {output_path}")


if __name__ == "__main__":
    main()
