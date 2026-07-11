"""
Split Dataset

Script to split data into train/dev/test sets:
- Group-based splitting to avoid entity leakage
- Stratified splitting by entity type
- Configurable ratios

Usage:
    python scripts/split_dataset.py --input data/synthetic/template_samples.jsonl
    python scripts/split_dataset.py --input data/synthetic/template_samples.jsonl --strategy stratified_group
    python scripts/split_dataset.py --input data/synthetic/template_samples.jsonl --train-ratio 0.7 --dev-ratio 0.15 --test-ratio 0.15
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splitter import Splitter, SplitConfig
from src.data.deduplicator import Deduplicator
from src.data.schema import load_samples, save_samples, save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/dev/test")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", "-o", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--strategy", "-s", type=str, default="stratified_group",
                       choices=["random", "stratified", "group", "stratified_group"],
                       help="Splitting strategy")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dedup-strategies", type=str, default="exact,entity_set",
                       help="Deduplication strategies (comma-separated)")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument("--report", type=str, help="Save report to file")
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Splitting")
    print("=" * 60)
    print(f"\nInput: {args.input}")
    print(f"Strategy: {args.strategy}")
    print(f"Ratios: train={args.train_ratio}, dev={args.dev_ratio}, test={args.test_ratio}")
    print(f"Seed: {args.seed}")

    # Load samples
    print("\nLoading samples...")
    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} samples")

    # Deduplicate
    if not args.skip_dedup:
        print("\nDeduplicating...")
        strategies = [s.strip() for s in args.dedup_strategies.split(',')]
        dedup = Deduplicator(strategies=strategies)
        result = dedup.deduplicate_and_report(samples)

        samples = result["unique_samples"]
        dedup_report = result["report"]

        print(f"Duplicates removed: {dedup_report['total_duplicates_removed']}")
        print(f"Unique samples: {len(samples)}")

    # Split
    print("\nSplitting...")
    config = SplitConfig(
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    splitter = Splitter(strategy=args.strategy)
    split_result = splitter.split_and_save(samples, config, args.output_dir)

    # Print statistics
    print("\n" + "=" * 60)
    print("Split Results")
    print("=" * 60)

    print(f"\nSample counts:")
    print(f"  Train: {split_result['train_count']} ({split_result['train_count']/split_result['total_count']:.1%})")
    print(f"  Dev: {split_result['dev_count']} ({split_result['dev_count']/split_result['total_count']:.1%})")
    print(f"  Test: {split_result['test_count']} ({split_result['test_count']/split_result['total_count']:.1%})")

    # Print entity distribution
    print(f"\nEntity distribution:")
    for split_name in ["train", "dev", "test"]:
        stats = split_result["statistics"][split_name]
        print(f"\n  {split_name}:")
        print(f"    Samples: {stats['sample_count']}")
        print(f"    Entities: {stats['entity_count']}")
        for etype, count in stats["entities_by_type"].items():
            # Replace unicode chars for Windows console compatibility
            safe_etype = etype.encode('ascii', 'replace').decode('ascii')
            print(f"    - {safe_etype}: {count}")

    # Print file paths
    print(f"\nOutput files:")
    print(f"  Train: {split_result['train_path']}")
    print(f"  Dev: {split_result['dev_path']}")
    print(f"  Test: {split_result['test_path']}")

    # Save report if requested
    if args.report:
        report_data = {
            "strategy": args.strategy,
            "config": {
                "train_ratio": args.train_ratio,
                "dev_ratio": args.dev_ratio,
                "test_ratio": args.test_ratio,
                "seed": args.seed,
            },
            **split_result,
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.report}")

    print("\n[OK] Split complete!")


if __name__ == "__main__":
    main()
