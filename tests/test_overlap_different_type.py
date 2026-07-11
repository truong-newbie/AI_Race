"""
Tests for Overlap Different Type

Test Rule 2: Same span, different types.
"""

import pytest
from src.entity.resolver import EntityResolver
from src.entity.confidence import ConfidenceConfig
from src.entity.conflict_logger import ConflictLogger


class TestOverlapDifferentType:
    """Test resolution for same span, different types."""

    @pytest.fixture
    def resolver(self):
        """Create resolver instance."""
        config = ConfidenceConfig()
        logger = ConflictLogger(enable_logging=False)
        return EntityResolver(config=config, conflict_logger=logger)

    def test_same_span_different_type_prefer_higher_confidence(self, resolver):
        """Test Rule 2: Prefer higher confidence when same span different type."""
        text = "viêm phổi"
        entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN", "confidence": 0.7, "source": "regex"},
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "TRIỆU_CHỨNG", "confidence": 0.9, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        assert resolved[0]["type"] == "TRIỆU_CHỨNG"  # Higher confidence

    def test_same_span_different_type_with_section_context(self, resolver):
        """Test Rule 6: Section context affects type preference."""
        text = "viêm phổi"
        entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN", "confidence": 0.8, "source": "ner"},
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "TRIỆU_CHỨNG", "confidence": 0.8, "source": "ner"},
        ]

        # In "chẩn đoán" section, should prefer CHẨN_ĐOÁN
        resolved = resolver.resolve_entities(entities, text, section="Chẩn đoán:")

        assert len(resolved) == 1
        assert resolved[0]["type"] == "CHẨN_ĐOÁN"

    def test_section_symptom_prefers_trieu_chung(self, resolver):
        """Test Rule 6: Symptom section prefers TRIỆU_CHỨNG."""
        text = "ho và sốt"
        entities = [
            {"text": "ho và sốt", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN", "confidence": 0.8, "source": "ner"},
            {"text": "ho và sốt", "start": 0, "end": 10, "type": "TRIỆU_CHỨNG", "confidence": 0.8, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text, section="Triệu chứng:")

        assert len(resolved) == 1
        assert resolved[0]["type"] == "TRIỆU_CHỨNG"

    def test_conflict_logged(self, resolver):
        """Test that type conflicts are logged."""
        text = "viêm phổi"
        entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN", "confidence": 0.7, "source": "regex"},
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "TRIỆU_CHỨNG", "confidence": 0.9, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        # Check conflict was logged
        report = resolver.logger.get_report()
        assert report.stats.get("same_span_different_type", 0) > 0

    def test_no_text_modification(self, resolver):
        """Test Rule 7: No new text is created."""
        text = "viêm phổi"
        entities = [
            {"text": "viêm", "start": 0, "end": 4, "type": "CHẨN_ĐOÁN", "confidence": 0.8, "source": "ner"},
            {"text": "phổi", "start": 5, "end": 9, "type": "TRIỆU_CHỨNG", "confidence": 0.8, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        # No text should be created - only existing spans
        for entity in resolved:
            assert entity["text"] in text

    def test_multiple_type_options(self, resolver):
        """Test with multiple type options."""
        text = "Paracetamol"
        entities = [
            {"text": "Paracetamol", "start": 0, "end": 12, "type": "THUỐC", "confidence": 0.7, "source": "regex"},
            {"text": "Paracetamol", "start": 0, "end": 12, "type": "THUỐC", "confidence": 0.8, "source": "dict"},
            {"text": "Paracetamol", "start": 0, "end": 12, "type": "TRIỆU_CHỨNG", "confidence": 0.6, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        assert resolved[0]["type"] == "THUỐC"  # Highest combined confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
