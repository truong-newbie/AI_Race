"""
Tests for Template Generator

Tests the template generator for:
- Sample generation
- Entity span alignment
- Entity type validation
- Assertion assignment
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.template_generator import TemplateGenerator
from src.data.schema import Sample, Entity


class TestTemplateGenerator:
    """Test TemplateGenerator."""

    @pytest.fixture
    def generator(self):
        return TemplateGenerator(seed=42)

    def test_generate_diagnosis(self, generator):
        """Test diagnosis sample generation."""
        samples = generator.generate_diagnosis(count=10)

        assert len(samples) == 10
        for sample in samples:
            assert isinstance(sample, Sample)
            assert sample.text
            assert len(sample.entities) >= 1
            assert sample.source == "template"

            # Check entity type
            for entity in sample.entities:
                assert entity.type == "CHẨN_ĐOÁN"
                assert entity.start >= 0
                assert entity.end > entity.start
                assert sample.text[entity.start:entity.end] == entity.text

    def test_generate_symptom(self, generator):
        """Test symptom sample generation."""
        samples = generator.generate_symptom(count=10)

        assert len(samples) == 10
        for sample in samples:
            assert isinstance(sample, Sample)
            assert sample.text
            assert len(sample.entities) >= 1

            for entity in sample.entities:
                assert entity.type == "TRIỆU_CHỨNG"
                assert sample.text[entity.start:entity.end] == entity.text

    def test_generate_drug(self, generator):
        """Test drug sample generation."""
        samples = generator.generate_drug(count=10)

        assert len(samples) == 10
        for sample in samples:
            assert isinstance(sample, Sample)
            assert sample.text
            assert len(sample.entities) >= 1

            for entity in sample.entities:
                assert entity.type == "THUỐC"
                assert sample.text[entity.start:entity.end] == entity.text
                # Drugs should have RxNorm candidates
                assert len(entity.candidates) >= 1

    def test_generate_lab(self, generator):
        """Test lab test sample generation."""
        samples = generator.generate_lab(count=10)

        assert len(samples) == 10
        for sample in samples:
            assert isinstance(sample, Sample)
            assert sample.text

            for entity in sample.entities:
                assert entity.type == "KẾT_QUẢ_XÉT_NGHIỆM"
                assert sample.text[entity.start:entity.end] == entity.text

    def test_generate_negated(self, generator):
        """Test negated sample generation."""
        samples = generator.generate_negated(count=10)

        assert len(samples) == 10
        for sample in samples:
            assert "isNegated" in sample.entities[0].assertions

    def test_generate_historical(self, generator):
        """Test historical sample generation."""
        samples = generator.generate_historical(count=10)

        assert len(samples) == 10
        for sample in samples:
            assert "isHistorical" in sample.entities[0].assertions

    def test_generate_family(self, generator):
        """Test family history sample generation."""
        samples = generator.generate_family(count=10)

        assert len(samples) == 10
        for sample in samples:
            assert "isFamily" in sample.entities[0].assertions

    def test_generate_multi_entity(self, generator):
        """Test multi-entity sample generation."""
        samples = generator.generate_multi_entity(count=5)

        assert len(samples) == 5
        for sample in samples:
            assert len(sample.entities) >= 2

    def test_generate_all(self, generator):
        """Test generating all types."""
        samples = generator.generate_all(count=100)

        assert len(samples) > 0

        # Count by type
        by_type = {}
        for sample in samples:
            for entity in sample.entities:
                by_type[entity.type] = by_type.get(entity.type, 0) + 1

        assert len(by_type) >= 4  # Should have multiple types
        assert sum(by_type.values()) > 0  # Should have entities

    def test_span_alignment(self, generator):
        """Test that all entity spans align with text."""
        samples = generator.generate_all(count=50)

        errors = []
        for sample in samples:
            for entity in sample.entities:
                try:
                    extracted = sample.text[entity.start:entity.end]
                    if extracted != entity.text:
                        errors.append({
                            "sample_id": sample.id,
                            "entity_text": entity.text,
                            "extracted": extracted,
                            "start": entity.start,
                            "end": entity.end,
                        })
                except IndexError:
                    errors.append({
                        "sample_id": sample.id,
                        "entity_text": entity.text,
                        "error": "IndexError",
                        "start": entity.start,
                        "end": entity.end,
                        "text_length": len(sample.text),
                    })

        assert len(errors) == 0, f"Span alignment errors: {errors[:5]}"

    def test_id_uniqueness(self, generator):
        """Test that sample IDs are unique."""
        samples = generator.generate_all(count=100)

        ids = [s.id for s in samples]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"

    def test_entity_positions(self, generator):
        """Test entity positions are valid."""
        samples = generator.generate_all(count=50)

        for sample in samples:
            for entity in sample.entities:
                assert entity.start >= 0
                assert entity.end > entity.start
                assert entity.end <= len(sample.text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
