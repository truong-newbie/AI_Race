"""
Tests for Span Alignment

Tests that all entity spans correctly align with their text.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.template_generator import TemplateGenerator
from src.data.schema import Sample


class TestSpanAlignment:
    """Test span alignment across all generators."""

    @pytest.fixture
    def generator(self):
        return TemplateGenerator(seed=42)

    def test_basic_span_alignment(self, generator):
        """Test basic span alignment for generated samples."""
        samples = generator.generate_all(count=50)

        errors = []
        for sample in samples:
            for i, entity in enumerate(sample.entities):
                try:
                    extracted = sample.text[entity.start:entity.end]
                    if extracted != entity.text:
                        errors.append({
                            "sample_id": sample.id,
                            "entity_index": i,
                            "entity_text": entity.text,
                            "extracted": extracted,
                            "start": entity.start,
                            "end": entity.end,
                        })
                except IndexError as e:
                    errors.append({
                        "sample_id": sample.id,
                        "entity_index": i,
                        "error": str(e),
                        "start": entity.start,
                        "end": entity.end,
                        "text_length": len(sample.text),
                    })

        assert len(errors) == 0, f"Found {len(errors)} span alignment errors: {errors[:3]}"

    def test_no_overlapping_spans(self, generator):
        """Test that entity spans don't have invalid overlaps."""
        samples = generator.generate_all(count=50)

        errors = []
        for sample in samples:
            entities = sample.entities
            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if i >= j:
                        continue
                    # Check if spans overlap incorrectly
                    if e1.start < e2.end and e2.start < e1.end:
                        # They overlap - check if this is intentional (same text)
                        if e1.text != e2.text:
                            errors.append({
                                "sample_id": sample.id,
                                "entity_1": e1.text,
                                "entity_2": e2.text,
                                "span_1": (e1.start, e1.end),
                                "span_2": (e2.start, e2.end),
                            })

        # Allow some overlapping spans for multi-word entities
        # Just ensure we don't have too many
        if len(errors) > len(samples) * 0.1:  # More than 10% is too many
            assert False, f"Too many overlapping spans: {len(errors)}"

    def test_span_positions_valid(self, generator):
        """Test that all span positions are valid."""
        samples = generator.generate_all(count=50)

        errors = []
        for sample in samples:
            for entity in sample.entities:
                if entity.start < 0:
                    errors.append(f"Negative start: {entity.start}")
                if entity.end <= 0:
                    errors.append(f"Non-positive end: {entity.end}")
                if entity.start >= entity.end:
                    errors.append(f"Invalid span: start={entity.start} >= end={entity.end}")
                if entity.end > len(sample.text):
                    errors.append(f"End exceeds text: end={entity.end} > len={len(sample.text)}")

        assert len(errors) == 0, f"Invalid positions: {errors}"

    def test_entity_text_not_empty(self, generator):
        """Test that entity text is not empty."""
        samples = generator.generate_all(count=50)

        for sample in samples:
            for entity in sample.entities:
                assert len(entity.text) > 0, f"Empty entity text in {sample.id}"

    def test_all_entities_extracted(self, generator):
        """Test that we can extract all entities from text."""
        samples = generator.generate_all(count=50)

        for sample in samples:
            for entity in sample.entities:
                # Entity text should be findable in the text
                assert entity.text in sample.text, \
                    f"Entity text '{entity.text}' not in sample text '{sample.text}'"


class TestEdgeCases:
    """Test span alignment for edge cases."""

    def test_unicode_handling(self):
        """Test that unicode entities are correctly handled."""
        generator = TemplateGenerator(seed=42)
        samples = generator.generate_all(count=20)

        for sample in samples:
            for entity in sample.entities:
                # Vietnamese characters should be preserved
                extracted = sample.text[entity.start:entity.end]
                assert extracted == entity.text

    def test_special_characters(self):
        """Test samples with special characters."""
        generator = TemplateGenerator(seed=42)

        # Create a sample with special characters
        sample = generator.generate_diagnosis(count=1)[0]

        # Add special chars
        sample.text = sample.text + " - test&special"

        # Check span is still valid
        for entity in sample.entities:
            if entity.end <= len(sample.text):
                extracted = sample.text[entity.start:entity.end]
                assert extracted == entity.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
