"""
Tests for Split No Leakage

Tests that the splitter doesn't cause data leakage:
- Group splitting prevents same entities in train/test
- No text overlap between splits
- Stratified splits maintain entity type distribution
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splitter import Splitter, SplitConfig, GroupSplitter, StratifiedSplitter
from src.data.schema import Sample, Entity


def create_sample(sample_id: str, text: str, entities: list = None, source: str = "test") -> Sample:
    """Helper to create a sample."""
    if entities is None:
        entities = [
            Entity(text="test", start=0, end=4, type="TRIỆU_CHỨNG", assertions=[], candidates=[])
        ]
    return Sample(id=sample_id, text=text, entities=entities, source=source)


class TestNoTextLeakage:
    """Test that there's no text leakage between splits."""

    def test_no_text_overlap_train_dev(self):
        """Test that train and dev don't share text."""
        samples = [
            create_sample(f"s{i}", f"Bệnh nhân ho {i}.")
            for i in range(50)
        ]

        config = SplitConfig(train_ratio=0.8, dev_ratio=0.1, seed=42)
        splitter = Splitter(strategy="random")
        result = splitter.split(samples, config)

        train_texts = {s.text for s in result.train}
        dev_texts = {s.text for s in result.dev}

        overlap = train_texts & dev_texts
        assert len(overlap) == 0, f"Text overlap found: {overlap}"

    def test_no_text_overlap_train_test(self):
        """Test that train and test don't share text."""
        samples = [
            create_sample(f"s{i}", f"Bệnh nhân ho {i}.")
            for i in range(50)
        ]

        config = SplitConfig(train_ratio=0.8, dev_ratio=0.1, seed=42)
        splitter = Splitter(strategy="random")
        result = splitter.split(samples, config)

        train_texts = {s.text for s in result.train}
        test_texts = {s.text for s in result.internal_test}

        overlap = train_texts & test_texts
        assert len(overlap) == 0, f"Text overlap found: {overlap}"

    def test_no_text_overlap_dev_test(self):
        """Test that dev and test don't share text."""
        samples = [
            create_sample(f"s{i}", f"Bệnh nhân ho {i}.")
            for i in range(50)
        ]

        config = SplitConfig(train_ratio=0.8, dev_ratio=0.1, seed=42)
        splitter = Splitter(strategy="random")
        result = splitter.split(samples, config)

        dev_texts = {s.text for s in result.dev}
        test_texts = {s.text for s in result.internal_test}

        overlap = dev_texts & test_texts
        assert len(overlap) == 0, f"Text overlap found: {overlap}"


class TestGroupSplitting:
    """Test group-based splitting."""

    def test_grouped_samples_stay_together(self):
        """Test that samples with shared entities are grouped."""
        # Create samples with shared entities
        shared_entity = Entity(text="viêm phổi", start=15, end=26, type="CHẨN_ĐOÁN", assertions=[], candidates=["J18.9"])

        samples = [
            create_sample("s1", "Bệnh nhân ho.", [
                Entity(text="ho", start=12, end=14, type="TRIỆU_CHỨNG", assertions=[], candidates=[])
            ]),
            create_sample("s2", "Chẩn đoán viêm phổi.", [
                shared_entity
            ]),
            create_sample("s3", "Tiền sử viêm phổi.", [
                shared_entity
            ]),
            create_sample("s4", "Bệnh nhân sốt.", [
                Entity(text="sốt", start=12, end=16, type="TRIỆU_CHỨNG", assertions=[], candidates=[])
            ]),
        ]

        config = SplitConfig(train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, seed=42)
        splitter = Splitter(strategy="group")
        result = splitter.split(samples, config)

        # Find which split each sample is in
        s1_split = None
        s2_split = None
        s3_split = None

        for name, split in [("train", result.train), ("dev", result.dev), ("test", result.internal_test)]:
            for s in split:
                if s.id == "s1":
                    s1_split = name
                if s.id == "s2":
                    s2_split = name
                if s.id == "s3":
                    s3_split = name

        # s2 and s3 share the same entity, so they should be in the same split
        assert s2_split == s3_split, "Samples with shared entities should be in same split"

    def test_stratified_distribution(self):
        """Test that stratified splitting maintains entity type distribution."""
        samples = [
            create_sample(f"s{i}", f"Text {i}.", [
                Entity(text="X", start=0, end=1, type="TRIỆU_CHỨNG", assertions=[], candidates=[])
            ])
            for i in range(40)
        ] + [
            create_sample(f"s{i}", f"Text {i}.", [
                Entity(text="X", start=0, end=1, type="CHẨN_ĐOÁN", assertions=[], candidates=[])
            ])
            for i in range(40, 80)
        ] + [
            create_sample(f"s{i}", f"Text {i}.", [
                Entity(text="X", start=0, end=1, type="THUỐC", assertions=[], candidates=[])
            ])
            for i in range(80, 100)
        ]

        config = SplitConfig(train_ratio=0.8, seed=42)
        splitter = Splitter(strategy="stratified")
        result = splitter.split(samples, config)

        # Count entity types in train
        train_types = {}
        for s in result.train:
            if s.entities:
                t = s.entities[0].type
                train_types[t] = train_types.get(t, 0) + 1

        # Should have all types represented
        assert len(train_types) == 3
        # Should be roughly proportional (80% each)
        assert train_types.get("TRIỆU_CHỨNG", 0) >= 30  # At least 30 of 40
        assert train_types.get("CHẨN_ĐOÁN", 0) >= 30
        assert train_types.get("THUỐC", 0) >= 14  # At least 14 of 20


class TestSplitRatios:
    """Test split ratio compliance."""

    def test_basic_ratios(self):
        """Test that splits follow configured ratios."""
        samples = [create_sample(f"s{i}", f"Text {i}.") for i in range(100)]

        config = SplitConfig(train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, seed=42)
        splitter = Splitter(strategy="random")
        result = splitter.split(samples, config)

        assert len(result.train) == 80
        assert len(result.dev) == 10
        assert len(result.internal_test) == 10

    def test_exact_ratios_large_dataset(self):
        """Test ratios with large dataset."""
        samples = [create_sample(f"s{i}", f"Text {i}.") for i in range(1000)]

        config = SplitConfig(train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, seed=42)
        splitter = Splitter(strategy="random")
        result = splitter.split(samples, config)

        assert len(result.train) == 700
        assert len(result.dev) == 150
        assert len(result.internal_test) == 150

    def test_custom_ratios(self):
        """Test with custom ratios."""
        samples = [create_sample(f"s{i}", f"Text {i}.") for i in range(100)]

        config = SplitConfig(train_ratio=0.6, dev_ratio=0.3, test_ratio=0.1, seed=42)
        splitter = Splitter(strategy="random")
        result = splitter.split(samples, config)

        assert len(result.train) == 60
        assert len(result.dev) == 30
        assert len(result.internal_test) == 10


class TestEdgeCases:
    """Test edge cases for splitting."""

    def test_small_dataset(self):
        """Test splitting with small dataset."""
        samples = [create_sample(f"s{i}", f"Text {i}.") for i in range(5)]

        config = SplitConfig(train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2, seed=42)
        splitter = Splitter(strategy="random")
        result = splitter.split(samples, config)

        # Should not crash
        assert len(result.train) + len(result.dev) + len(result.internal_test) == 5

    def test_all_same_text(self):
        """Test with all same text samples."""
        samples = [create_sample(f"s{i}", "Same text.") for i in range(10)]

        config = SplitConfig(train_ratio=0.8, dev_ratio=0.1, seed=42)
        splitter = Splitter(strategy="group")
        result = splitter.split(samples, config)

        # Should still split (even if content is same)
        total = len(result.train) + len(result.dev) + len(result.internal_test)
        assert total == 10

    def test_empty_entities(self):
        """Test splitting samples with empty entities."""
        samples = [
            create_sample(f"s{i}", f"Text {i}.", [])
            for i in range(20)
        ]

        config = SplitConfig(train_ratio=0.8, seed=42)
        splitter = Splitter(strategy="random")
        result = splitter.split(samples, config)

        assert len(result.train) == 16
        assert len(result.dev) == 2
        assert len(result.internal_test) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
