"""
Train/Dev/Test Splitter

Split data into train/dev/test sets:
- Group-based splitting to avoid entity leakage
- Stratified splitting by entity type
- Configurable ratios
"""

import random
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .schema import Sample, Entity, DatasetSplit


# =============================================================================
# Split Configuration
# =============================================================================

@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    stratify_by: str = "entity_type"  # entity_type, source, or None
    group_by: Optional[str] = None  # group entities to avoid leakage

    def __post_init__(self):
        total = self.train_ratio + self.dev_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


# =============================================================================
# Splitting Strategies
# =============================================================================

class BaseSplitter:
    """Base class for data splitters."""

    def split(self, samples: List[Sample], config: SplitConfig) -> DatasetSplit:
        raise NotImplementedError


class RandomSplitter(BaseSplitter):
    """
    Random splitting without stratification.

    Simple random split based on ratios.
    """

    def split(self, samples: List[Sample], config: SplitConfig) -> DatasetSplit:
        random.seed(config.seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * config.train_ratio)
        dev_end = train_end + int(n * config.dev_ratio)

        train_samples = shuffled[:train_end]
        dev_samples = shuffled[train_end:dev_end]
        test_samples = shuffled[dev_end:]

        return DatasetSplit(
            train=train_samples,
            dev=dev_samples,
            internal_test=test_samples,
        )


class StratifiedSplitter(BaseSplitter):
    """
    Stratified splitting by entity type.

    Ensures each entity type is proportionally represented in each split.
    """

    def split(self, samples: List[Sample], config: SplitConfig) -> DatasetSplit:
        random.seed(config.seed)

        # Group samples by primary entity type
        by_type: Dict[str, List[Sample]] = defaultdict(list)
        for sample in samples:
            if sample.entities:
                # Use first entity's type as the sample's type
                primary_type = sample.entities[0].type
                by_type[primary_type].append(sample)
            else:
                by_type["no_entity"].append(sample)

        train_samples = []
        dev_samples = []
        test_samples = []

        for etype, type_samples in by_type.items():
            # Shuffle within type
            shuffled = type_samples.copy()
            random.shuffle(shuffled)

            n = len(shuffled)
            train_end = int(n * config.train_ratio)
            dev_end = train_end + int(n * config.dev_ratio)

            train_samples.extend(shuffled[:train_end])
            dev_samples.extend(shuffled[train_end:dev_end])
            test_samples.extend(shuffled[dev_end:])

        return DatasetSplit(
            train=train_samples,
            dev=dev_samples,
            internal_test=test_samples,
        )


class GroupSplitter(BaseSplitter):
    """
    Group-based splitting to avoid entity leakage.

    Groups samples by shared entities and ensures groups are not split
    across train/dev/test. This prevents the model from seeing
    entities during training that are related to entities in test.
    """

    def split(self, samples: List[Sample], config: SplitConfig) -> DatasetSplit:
        random.seed(config.seed)

        # Build entity groups
        entity_to_groups: Dict[str, Set[int]] = defaultdict(set)  # entity_text -> group_ids
        sample_to_groups: Dict[int, Set[str]] = defaultdict(set)  # sample_idx -> entity_texts

        for i, sample in enumerate(samples):
            entity_texts = set(e.text for e in sample.entities)
            for et in entity_texts:
                entity_to_groups[et].add(i)
            sample_to_groups[i] = entity_texts

        # Build groups using union-find approach
        group_id = 0
        sample_group: Dict[int, int] = {}

        for i in range(len(samples)):
            if i in sample_group:
                continue

            # BFS to find all related samples
            queue = [i]
            visited = {i}
            current_group = {i}

            while queue:
                sample_idx = queue.pop(0)
                for entity_text in sample_to_groups[sample_idx]:
                    for related_idx in entity_to_groups[entity_text]:
                        if related_idx not in visited:
                            visited.add(related_idx)
                            queue.append(related_idx)
                            current_group.add(related_idx)

            # Assign group ID to all samples in this group
            for sample_idx in current_group:
                sample_group[sample_idx] = group_id
            group_id += 1

        # Group samples by their group ID
        groups: Dict[int, List[Tuple[int, Sample]]] = defaultdict(list)
        for i, sample in enumerate(samples):
            gid = sample_group[i]
            groups[gid].append((i, sample))

        # Convert to list for shuffling
        group_list = list(groups.values())
        random.shuffle(group_list)

        # Split groups
        n_groups = len(group_list)
        train_end = int(n_groups * config.train_ratio)
        dev_end = train_end + int(n_groups * config.dev_ratio)

        train_indices = set()
        dev_indices = set()
        test_indices = set()

        for i, group in enumerate(group_list):
            indices = {idx for idx, _ in group}
            if i < train_end:
                train_indices.update(indices)
            elif i < dev_end:
                dev_indices.update(indices)
            else:
                test_indices.update(indices)

        # Build result
        train_samples = [samples[i] for i in sorted(train_indices)]
        dev_samples = [samples[i] for i in sorted(dev_indices)]
        test_samples = [samples[i] for i in sorted(test_indices)]

        return DatasetSplit(
            train=train_samples,
            dev=dev_samples,
            internal_test=test_samples,
        )


class StratifiedGroupSplitter(BaseSplitter):
    """
    Stratified group-based splitting.

    Combines stratification by entity type with group-based splitting.
    """

    def split(self, samples: List[Sample], config: SplitConfig) -> DatasetSplit:
        random.seed(config.seed)

        # Group samples by primary entity type
        by_type: Dict[str, List[int]] = defaultdict(list)  # type -> sample_indices
        for i, sample in enumerate(samples):
            if sample.entities:
                primary_type = sample.entities[0].type
                by_type[primary_type].append(i)
            else:
                by_type["no_entity"].append(i)

        train_samples = []
        dev_samples = []
        test_samples = []

        for etype, indices in by_type.items():
            type_samples = [samples[i] for i in indices]

            # Within each type, use group-based splitting
            splitter = GroupSplitter()
            type_config = SplitConfig(
                train_ratio=config.train_ratio,
                dev_ratio=config.dev_ratio,
                test_ratio=config.test_ratio,
                seed=config.seed + hash(etype) % 1000,  # Different seed per type
            )

            split_result = splitter.split(type_samples, type_config)
            train_samples.extend(split_result.train)
            dev_samples.extend(split_result.dev)
            test_samples.extend(split_result.internal_test)

        return DatasetSplit(
            train=train_samples,
            dev=dev_samples,
            internal_test=test_samples,
        )


# =============================================================================
# Splitter Factory
# =============================================================================

class Splitter:
    """
    Main splitter class.

    Usage:
        splitter = Splitter()
        split = splitter.split(samples, config)
    """

    SPLITTERS = {
        'random': RandomSplitter,
        'stratified': StratifiedSplitter,
        'group': GroupSplitter,
        'stratified_group': StratifiedGroupSplitter,
    }

    def __init__(self, strategy: str = "stratified_group"):
        """
        Initialize splitter.

        Args:
            strategy: Splitting strategy
                - 'random': Simple random split
                - 'stratified': Stratified by entity type
                - 'group': Group-based to avoid leakage
                - 'stratified_group': Stratified + group-based
        """
        self.strategy = strategy
        if strategy not in self.SPLITTERS:
            raise ValueError(f"Unknown strategy: {strategy}. Options: {list(self.SPLITTERS.keys())}")

    def split(self, samples: List[Sample], config: Optional[SplitConfig] = None) -> DatasetSplit:
        """
        Split samples.

        Args:
            samples: List of samples to split
            config: Split configuration (uses defaults if not provided)

        Returns:
            DatasetSplit with train/dev/test samples
        """
        if config is None:
            config = SplitConfig()

        splitter_class = self.SPLITTERS[self.strategy]
        splitter = splitter_class()
        return splitter.split(samples, config)

    def split_and_save(
        self,
        samples: List[Sample],
        config: Optional[SplitConfig] = None,
        output_dir: str = "data/processed",
    ) -> Dict[str, Any]:
        """
        Split and save to files.

        Args:
            samples: List of samples
            config: Split configuration
            output_dir: Output directory

        Returns:
            Dictionary with split statistics
        """
        split_result = self.split(samples, config)

        # Save each split
        from .schema import save_jsonl

        train_path = f"{output_dir}/train.jsonl"
        dev_path = f"{output_dir}/dev.jsonl"
        test_path = f"{output_dir}/internal_test.jsonl"

        save_jsonl(train_path, [s.to_dict() for s in split_result.train])
        save_jsonl(dev_path, [s.to_dict() for s in split_result.dev])
        save_jsonl(test_path, [s.to_dict() for s in split_result.internal_test])

        # Compute statistics
        stats = self.compute_statistics(split_result)

        return {
            "train_count": len(split_result.train),
            "dev_count": len(split_result.dev),
            "test_count": len(split_result.internal_test),
            "total_count": len(samples),
            "train_path": train_path,
            "dev_path": dev_path,
            "test_path": test_path,
            "statistics": stats,
        }

    def compute_statistics(self, split: DatasetSplit) -> Dict[str, Any]:
        """Compute statistics for a split."""
        stats = {}

        for split_name, samples in [("train", split.train), ("dev", split.dev), ("test", split.internal_test)]:
            entity_counts: Dict[str, int] = defaultdict(int)
            total_entities = 0

            for sample in samples:
                for entity in sample.entities:
                    entity_counts[entity.type] += 1
                    total_entities += 1

            stats[split_name] = {
                "sample_count": len(samples),
                "entity_count": total_entities,
                "entities_by_type": dict(entity_counts),
            }

        return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for splitter."""
    import argparse
    from .schema import load_samples

    parser = argparse.ArgumentParser(description="Split data into train/dev/test")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", "-o", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--strategy", "-s", type=str, default="stratified_group",
                       choices=["random", "stratified", "group", "stratified_group"],
                       help="Splitting strategy")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load samples
    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} samples")

    # Configure split
    config = SplitConfig(
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Split
    splitter = Splitter(strategy=args.strategy)
    result = splitter.split_and_save(samples, config, args.output_dir)

    # Print report
    print("\n" + "=" * 60)
    print("Split Report")
    print("=" * 60)
    print(f"\nStrategy: {args.strategy}")
    print(f"\nSplit sizes:")
    print(f"  Train: {result['train_count']} ({result['train_count']/result['total_count']:.1%})")
    print(f"  Dev: {result['dev_count']} ({result['dev_count']/result['total_count']:.1%})")
    print(f"  Test: {result['test_count']} ({result['test_count']/result['total_count']:.1%})")

    print(f"\nEntity distribution:")
    for split_name in ["train", "dev", "test"]:
        stats = result["statistics"][split_name]
        print(f"\n  {split_name}:")
        for etype, count in stats["entities_by_type"].items():
            print(f"    {etype}: {count}")

    print(f"\nFiles saved:")
    print(f"  Train: {result['train_path']}")
    print(f"  Dev: {result['dev_path']}")
    print(f"  Test: {result['test_path']}")


if __name__ == "__main__":
    main()
