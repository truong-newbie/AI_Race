"""
Data Deduplicator

Deduplicate samples based on:
- Exact text match
- Entity set similarity
- Semantic similarity (optional)
"""

import hashlib
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .schema import Sample, Entity


# =============================================================================
# Hash-based Deduplication
# =============================================================================

class TextHasher:
    """Generate hashes for text deduplication."""

    @staticmethod
    def hash_text(text: str) -> str:
        """Hash full text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def hash_normalized(text: str) -> str:
        """Hash normalized text (lowercase, no diacritics)."""
        normalized = text.lower()
        # Remove diacritics approximation
        normalized = normalized.replace('ă', 'a').replace('â', 'a').replace('đ', 'd')
        normalized = normalized.replace('ê', 'e').replace('ô', 'o').replace('ơ', 'o').replace('ư', 'u')
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# =============================================================================
# Entity-based Deduplication
# =============================================================================

class EntityHasher:
    """Generate hashes for entity-based deduplication."""

    @staticmethod
    def hash_entities(entities: List[Entity]) -> str:
        """Hash entity set (ignoring positions)."""
        entity_strs = []
        for e in sorted(entities, key=lambda x: (x.start, x.end)):
            # Create entity signature without position
            sig = f"{e.text}|{e.type}|{','.join(sorted(e.assertions))}"
            entity_strs.append(sig)
        return hashlib.sha256(';'.join(entity_strs).encode()).hexdigest()[:16]

    @staticmethod
    def hash_entity_spans(entities: List[Entity]) -> str:
        """Hash entity spans (including positions)."""
        entity_strs = []
        for e in sorted(entities, key=lambda x: (x.start, x.end)):
            sig = f"{e.text}|{e.type}|{e.start}|{e.end}|{','.join(sorted(e.assertions))}"
            entity_strs.append(sig)
        return hashlib.sha256(';'.join(entity_strs).encode()).hexdigest()[:16]


# =============================================================================
# Deduplication Strategies
# =============================================================================

@dataclass
class DeduplicationResult:
    """Kết quả deduplication."""
    original_count: int
    duplicate_count: int
    unique_count: int
    duplicates: List[str] = field(default_factory=list)  # IDs of duplicates removed

    def summary(self) -> Dict[str, Any]:
        return {
            "original_count": self.original_count,
            "duplicate_count": self.duplicate_count,
            "unique_count": self.unique_count,
            "duplicate_rate": self.duplicate_count / self.original_count if self.original_count > 0 else 0,
            "duplicates_removed": self.duplicates[:100],  # Limit to first 100
        }


class ExactTextDeduplicator:
    """
    Deduplicate by exact text match.

    Keeps the first occurrence of each unique text.
    """

    def deduplicate(self, samples: List[Sample]) -> Tuple[List[Sample], DeduplicationResult]:
        seen_texts: Dict[str, str] = {}  # text_hash -> sample_id
        seen_text_full: Dict[str, str] = {}  # full_text -> sample_id
        duplicates = []
        unique_samples = []

        for sample in samples:
            text_hash = TextHasher.hash_text(sample.text)

            if text_hash in seen_texts:
                duplicates.append(sample.id)
                continue

            if sample.text in seen_text_full:
                duplicates.append(sample.id)
                continue

            seen_texts[text_hash] = sample.id
            seen_text_full[sample.text] = sample.id
            unique_samples.append(sample)

        return unique_samples, DeduplicationResult(
            original_count=len(samples),
            duplicate_count=len(duplicates),
            unique_count=len(unique_samples),
            duplicates=duplicates,
        )


class NormalizedTextDeduplicator:
    """
    Deduplicate by normalized text match.

    Text is normalized by lowercasing and removing diacritics.
    Keeps the first occurrence.
    """

    def deduplicate(self, samples: List[Sample]) -> Tuple[List[Sample], DeduplicationResult]:
        seen_normalized: Dict[str, str] = {}
        duplicates = []
        unique_samples = []

        for sample in samples:
            norm_hash = TextHasher.hash_normalized(sample.text)

            if norm_hash in seen_normalized:
                duplicates.append(sample.id)
                continue

            seen_normalized[norm_hash] = sample.id
            unique_samples.append(sample)

        return unique_samples, DeduplicationResult(
            original_count=len(samples),
            duplicate_count=len(duplicates),
            unique_count=len(unique_samples),
            duplicates=duplicates,
        )


class EntitySetDeduplicator:
    """
    Deduplicate by entity set match (ignoring positions).

    Two samples are considered duplicates if they have the same
    entity types and texts (but possibly different positions).
    """

    def deduplicate(self, samples: List[Sample]) -> Tuple[List[Sample], DeduplicationResult]:
        seen_entity_hashes: Dict[str, str] = {}
        duplicates = []
        unique_samples = []

        for sample in samples:
            entity_hash = EntityHasher.hash_entities(sample.entities)

            if entity_hash in seen_entity_hashes:
                duplicates.append(sample.id)
                continue

            seen_entity_hashes[entity_hash] = sample.id
            unique_samples.append(sample)

        return unique_samples, DeduplicationResult(
            original_count=len(samples),
            duplicate_count=len(duplicates),
            unique_count=len(unique_samples),
            duplicates=duplicates,
        )


class EntitySpanDeduplicator:
    """
    Deduplicate by entity spans (including positions).

    This is stricter than EntitySetDeduplicator - two samples
    are duplicates only if entities are at the same positions.
    """

    def deduplicate(self, samples: List[Sample]) -> Tuple[List[Sample], DeduplicationResult]:
        seen_span_hashes: Dict[str, str] = {}
        duplicates = []
        unique_samples = []

        for sample in samples:
            span_hash = EntityHasher.hash_entity_spans(sample.entities)

            if span_hash in seen_span_hashes:
                duplicates.append(sample.id)
                continue

            seen_span_hashes[span_hash] = sample.id
            unique_samples.append(sample)

        return unique_samples, DeduplicationResult(
            original_count=len(samples),
            duplicate_count=len(duplicates),
            unique_count=len(unique_samples),
            duplicates=duplicates,
        )


# =============================================================================
# Composite Deduplicator
# =============================================================================

class Deduplicator:
    """
    Multi-strategy deduplicator.

    Usage:
        dedup = Deduplicator()
        unique_samples, result = dedup.deduplicate(samples)
    """

    def __init__(self, strategies: Optional[List[str]] = None):
        """
        Initialize deduplicator.

        Args:
            strategies: List of strategies to use in order.
                Options: 'exact', 'normalized', 'entity_set', 'entity_span'
                Default: ['exact', 'entity_set']
        """
        self.strategies = strategies or ['exact', 'entity_set']
        self.deduplicators = {
            'exact': ExactTextDeduplicator(),
            'normalized': NormalizedTextDeduplicator(),
            'entity_set': EntitySetDeduplicator(),
            'entity_span': EntitySpanDeduplicator(),
        }

    def deduplicate(self, samples: List[Sample]) -> Tuple[List[Sample], List[DeduplicationResult]]:
        """
        Deduplicate samples using configured strategies.

        Args:
            samples: List of samples to deduplicate

        Returns:
            Tuple of (unique_samples, list of results per strategy)
        """
        results = []
        current_samples = samples

        for strategy_name in self.strategies:
            if strategy_name not in self.deduplicators:
                continue

            dedup = self.deduplicators[strategy_name]
            unique, result = dedup.deduplicate(current_samples)
            results.append(result)
            current_samples = unique

        return current_samples, results

    def deduplicate_and_report(self, samples: List[Sample]) -> Dict[str, Any]:
        """
        Deduplicate and generate detailed report.

        Args:
            samples: List of samples

        Returns:
            Dictionary with results and report
        """
        unique, results = self.deduplicate(samples)

        total_original = results[0].original_count if results else len(samples)
        total_duplicates = sum(r.duplicate_count for r in results)
        total_unique = len(unique)

        report = {
            "total_original": total_original,
            "total_duplicates_removed": total_duplicates,
            "total_unique": total_unique,
            "overall_deduplication_rate": total_duplicates / total_original if total_original > 0 else 0,
            "strategy_results": [r.summary() for r in results],
            "strategy_order": self.strategies,
        }

        return {
            "unique_samples": unique,
            "report": report,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for deduplicator."""
    import argparse
    from .schema import load_samples, save_samples

    parser = argparse.ArgumentParser(description="Deduplicate synthetic data")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--strategies", "-s", type=str, default="exact,entity_set",
                       help="Deduplication strategies (comma-separated): exact, normalized, entity_set, entity_span")
    parser.add_argument("--report", "-r", type=str, help="Save report to file")
    args = parser.parse_args()

    # Parse strategies
    strategies = [s.strip() for s in args.strategies.split(',')]

    # Load samples
    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} samples")

    # Deduplicate
    dedup = Deduplicator(strategies=strategies)
    result = dedup.deduplicate_and_report(samples)

    # Print report
    print("\n" + "=" * 60)
    print("Deduplication Report")
    print("=" * 60)
    print(f"\nOriginal samples: {result['report']['total_original']}")
    print(f"Duplicates removed: {result['report']['total_duplicates_removed']}")
    print(f"Unique samples: {result['report']['total_unique']}")
    print(f"Deduplication rate: {result['report']['overall_deduplication_rate']:.2%}")

    print(f"\nStrategies applied: {result['report']['strategy_order']}")
    for i, sr in enumerate(result['report']['strategy_results']):
        print(f"  {i+1}. {result['report']['strategy_order'][i]}: {sr['duplicate_count']} duplicates removed")

    # Save unique samples
    save_samples(args.output, result['unique_samples'])
    print(f"\nSaved {len(result['unique_samples'])} unique samples to {args.output}")

    # Save report if requested
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result['report'], f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
