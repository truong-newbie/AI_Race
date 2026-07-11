"""
Tests for Exact Duplicate Merge

Test Rule 1: Same span, same type should be merged.
"""

import pytest
from src.entity.resolver import EntityResolver
from src.entity.confidence import ConfidenceConfig
from src.entity.conflict_logger import ConflictLogger


class TestExactDuplicateMerge:
    """Test exact duplicate entity merging."""

    @pytest.fixture
    def resolver(self):
        """Create resolver instance."""
        config = ConfidenceConfig()
        logger = ConflictLogger(enable_logging=False)
        return EntityResolver(config=config, conflict_logger=logger)

    def test_merge_exact_duplicates_same_source(self, resolver):
        """Test merging entities from same source with same span and type."""
        text = "Bệnh nhân bị viêm phổi."
        # Calculate correct spans
        viêm_phổi_start = text.index("viêm phổi")
        viêm_phổi_end = viêm_phổi_start + len("viêm phổi")
        entities = [
            {"text": "viêm phổi", "start": viêm_phổi_start, "end": viêm_phổi_end, "type": "CHẨN_ĐOÁN", "confidence": 0.9, "source": "regex"},
            {"text": "viêm phổi", "start": viêm_phổi_start, "end": viêm_phổi_end, "type": "CHẨN_ĐOÁN", "confidence": 0.85, "source": "ner_model"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        assert resolved[0]["type"] == "CHẨN_ĐOÁN"
        assert "viêm" in resolved[0]["text"] and "phổi" in resolved[0]["text"]
        assert resolved[0]["confidence"] > 0.85  # Should be boosted
        assert "," in resolved[0]["source"]  # Multiple sources recorded

    def test_merge_exact_duplicates_different_sources(self, resolver):
        """Test merging entities from different sources."""
        text = "Bệnh nhân bị viêm phổi."
        viêm_phổi_start = text.index("viêm phổi")
        viêm_phổi_end = viêm_phổi_start + len("viêm phổi")
        entities = [
            {"text": "viêm phổi", "start": viêm_phổi_start, "end": viêm_phổi_end, "type": "CHẨN_ĐOÁN", "confidence": 0.9, "source": "regex"},
            {"text": "viêm phổi", "start": viêm_phổi_start, "end": viêm_phổi_end, "type": "CHẨN_ĐOÁN", "confidence": 0.8, "source": "ner_model"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        assert "regex" in resolved[0]["source"]
        assert "ner_model" in resolved[0]["source"]
        assert resolved[0]["confidence"] > 0.8  # Agreement bonus

    def test_keep_different_spans_same_type(self, resolver):
        """Test that different spans are kept separate."""
        text = "ho và sốt"
        ho_start = text.index("ho")
        sốt_start = text.index("sốt")
        entities = [
            {"text": "ho", "start": ho_start, "end": ho_start + 2, "type": "TRIỆU_CHỨNG", "confidence": 0.9, "source": "regex"},
            {"text": "sốt", "start": sốt_start, "end": sốt_start + 3, "type": "TRIỆU_CHỨNG", "confidence": 0.9, "source": "regex"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 2

    def test_merge_three_duplicates(self, resolver):
        """Test merging three identical entities."""
        text = "Paracetamol"
        entities = [
            {"text": "Paracetamol", "start": 0, "end": 12, "type": "THUỐC", "confidence": 0.9, "source": "regex"},
            {"text": "Paracetamol", "start": 0, "end": 12, "type": "THUỐC", "confidence": 0.85, "source": "dict"},
            {"text": "Paracetamol", "start": 0, "end": 12, "type": "THUỐC", "confidence": 0.8, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        assert "regex" in resolved[0]["source"]
        assert "dict" in resolved[0]["source"]
        assert "ner" in resolved[0]["source"]
        assert len(resolved[0]["source_scores"]) == 3

    def test_text_from_original(self, resolver):
        """Test that text is always extracted from original."""
        text = "Bệnh nhân bị viêm phổi."
        viêm_phổi_start = text.index("viêm phổi")
        viêm_phổi_end = viêm_phổi_start + len("viêm phổi")
        entities = [
            {"text": "WRONG", "start": viêm_phổi_start, "end": viêm_phổi_end, "type": "CHẨN_ĐOÁN", "confidence": 0.9, "source": "test"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        assert resolved[0]["text"] == "viêm phổi"  # From original text

    def test_empty_entity_list(self, resolver):
        """Test with empty entity list."""
        text = "Bệnh nhân bình thường."
        resolved = resolver.resolve_entities([], text)
        assert len(resolved) == 0

    def test_single_entity(self, resolver):
        """Test with single entity (no merge needed)."""
        text = "ho bệnh"
        ho_start = text.index("ho")
        entities = [
            {"text": "ho", "start": ho_start, "end": ho_start + 2, "type": "TRIỆU_CHỨNG", "confidence": 0.9, "source": "regex"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        assert resolved[0]["text"] == "ho"
        assert resolved[0]["source"] == "regex"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
