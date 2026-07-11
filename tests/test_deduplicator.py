"""
Tests for Deduplicator

Tests the deduplication functionality:
- Exact text deduplication
- Normalized text deduplication
- Entity set deduplication
- Entity span deduplication
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.deduplicator import (
    Deduplicator, ExactTextDeduplicator, NormalizedTextDeduplicator,
    EntitySetDeduplicator, EntitySpanDeduplicator
)
from src.data.schema import Sample, Entity


def create_sample(sample_id: str, text: str, entities: list = None) -> Sample:
    """Helper to create a sample."""
    if entities is None:
        entities = [
            Entity(text="test", start=0, end=4, type="TRIỆU_CHỨNG", assertions=[], candidates=[])
        ]
    return Sample(id=sample_id, text=text, entities=entities, source="test")


class TestDeduplicator:
    """Test Deduplicator class."""

    def test_exact_deduplication(self):
        """Test exact text deduplication."""
        samples = [
            create_sample("s1", "Bệnh nhân ho.", []),
            create_sample("s2", "Bệnh nhân ho.", []),
            create_sample("s3", "Bệnh nhân sốt.", []),
        ]

        dedup = Deduplicator(strategies=['exact'])
        unique, results = dedup.deduplicate(samples)

        assert len(unique) == 2
        assert results[0].duplicate_count == 1

    def test_normalized_deduplication(self):
        """Test normalized text deduplication."""
        samples = [
            create_sample("s1", "Bệnh nhân ho.", []),
            create_sample("s2", "Bệnh nhân Ho.", []),
            create_sample("s3", "Bệnh nhân ho", []),
            create_sample("s4", "Bệnh nhân sốt.", []),
        ]

        dedup = Deduplicator(strategies=['normalized'])
        unique, results = dedup.deduplicate(samples)

        assert len(unique) >= 2

    def test_entity_set_deduplication(self):
        """Test entity set deduplication."""
        entities1 = [
            Entity(text="ho", start=12, end=14, type="TRIỆU_CHỨNG", assertions=[], candidates=[])
        ]
        entities2 = [
            Entity(text="ho", start=12, end=14, type="TRIỆU_CHỨNG", assertions=[], candidates=[])
        ]
        entities3 = [
            Entity(text="sốt", start=12, end=16, type="TRIỆU_CHỨNG", assertions=[], candidates=[])
        ]

        samples = [
            create_sample("s1", "Bệnh nhân ho.", entities1),
            create_sample("s2", "BN ho.", entities2),
            create_sample("s3", "BN sốt.", entities3),
        ]

        dedup = Deduplicator(strategies=['entity_set'])
        unique, results = dedup.deduplicate(samples)

        # s1 and s2 should be duplicates (same entity set)
        assert len(unique) == 2

    def test_combined_strategies(self):
        """Test combined deduplication strategies."""
        samples = [
            create_sample("s1", "Bệnh nhân ho.", []),
            create_sample("s2", "Bệnh nhân ho.", []),
            create_sample("s3", "BN ho.", []),
            create_sample("s4", "Bệnh nhân sốt.", []),
        ]

        dedup = Deduplicator(strategies=['exact', 'entity_set'])
        unique, results = dedup.deduplicate(samples)

        # Should remove duplicates from all strategies
        assert len(unique) <= 2

    def test_no_duplicates(self):
        """Test with no duplicates."""
        samples = [
            create_sample("s1", "Bệnh nhân ho.", []),
            create_sample("s2", "Bệnh nhân sốt.", []),
            create_sample("s3", "Bệnh nhân đau.", []),
        ]

        dedup = Deduplicator(strategies=['exact'])
        unique, results = dedup.deduplicate(samples)

        assert len(unique) == 3
        assert results[0].duplicate_count == 0

    def test_all_duplicates(self):
        """Test with all duplicates."""
        samples = [
            create_sample("s1", "Bệnh nhân ho.", []),
            create_sample("s2", "Bệnh nhân ho.", []),
            create_sample("s3", "Bệnh nhân ho.", []),
        ]

        dedup = Deduplicator(strategies=['exact'])
        unique, results = dedup.deduplicate(samples)

        assert len(unique) == 1
        assert results[0].duplicate_count == 2

    def test_empty_list(self):
        """Test with empty list."""
        dedup = Deduplicator(strategies=['exact'])
        unique, results = dedup.deduplicate([])

        assert len(unique) == 0
        assert results[0].original_count == 0


class TestIndividualDeduplicators:
    """Test individual deduplicator classes."""

    def test_exact_text_deduplicator(self):
        """Test ExactTextDeduplicator."""
        dedup = ExactTextDeduplicator()
        samples = [
            create_sample("s1", "Text A", []),
            create_sample("s2", "Text A", []),
            create_sample("s3", "Text B", []),
        ]
        unique, result = dedup.deduplicate(samples)

        assert len(unique) == 2
        assert result.duplicate_count == 1

    def test_normalized_text_deduplicator(self):
        """Test NormalizedTextDeduplicator."""
        dedup = NormalizedTextDeduplicator()
        samples = [
            create_sample("s1", "Text A", []),
            create_sample("s2", "text a", []),
            create_sample("s3", "TEXT A", []),
        ]
        unique, result = dedup.deduplicate(samples)

        assert len(unique) == 1
        assert result.duplicate_count == 2

    def test_entity_set_deduplicator(self):
        """Test EntitySetDeduplicator."""
        dedup = EntitySetDeduplicator()
        entities = [
            Entity(text="X", start=0, end=1, type="T", assertions=[], candidates=[])
        ]
        samples = [
            create_sample("s1", "Different text", entities),
            create_sample("s2", "Also different", entities),
        ]
        unique, result = dedup.deduplicate(samples)

        assert len(unique) == 1

    def test_entity_span_deduplicator(self):
        """Test EntitySpanDeduplicator."""
        dedup = EntitySpanDeduplicator()

        # Same entity set but different spans
        entities1 = [Entity(text="X", start=0, end=1, type="T", assertions=[], candidates=[])]
        entities2 = [Entity(text="X", start=5, end=6, type="T", assertions=[], candidates=[])]

        samples = [
            create_sample("s1", "Text X here", entities1),
            create_sample("s2", "Other X here", entities2),
        ]
        unique, result = dedup.deduplicate(samples)

        # Different spans = not duplicates
        assert len(unique) == 2


class TestDeduplicationResult:
    """Test deduplication result."""

    def test_summary(self):
        """Test result summary."""
        dedup = ExactTextDeduplicator()
        samples = [
            create_sample("s1", "A", []),
            create_sample("s2", "A", []),
            create_sample("s3", "B", []),
        ]
        unique, result = dedup.deduplicate(samples)

        summary = result.summary()
        assert summary["original_count"] == 3
        assert summary["duplicate_count"] == 1
        assert summary["unique_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
