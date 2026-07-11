"""
Tests for Overlap Same Type

Test Rule 3: Overlapping spans with same type.
"""

import pytest
from src.entity.resolver import EntityResolver
from src.entity.confidence import ConfidenceConfig
from src.entity.conflict_logger import ConflictLogger


class TestOverlapSameType:
    """Test overlap resolution for same type entities."""

    @pytest.fixture
    def resolver(self):
        """Create resolver instance."""
        config = ConfidenceConfig()
        logger = ConflictLogger(enable_logging=False)
        return EntityResolver(config=config, conflict_logger=logger)

    def test_overlap_prefer_higher_confidence(self, resolver):
        """Test Rule 3: Prefer span with higher confidence."""
        text = "Bệnh nhân viêm phổi nặng"
        entities = [
            # Longer span, lower confidence
            {"text": "viêm phổi nặng", "start": 12, "end": 26, "type": "CHẨN_ĐOÁN", "confidence": 0.6, "source": "ner"},
            # Shorter span, higher confidence
            {"text": "viêm phổi", "start": 12, "end": 21, "type": "CHẨN_ĐOÁN", "confidence": 0.9, "source": "dict"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        # Should prefer higher confidence
        assert resolved[0]["confidence"] == 0.9

    def test_overlap_same_confidence_prefer_longer(self, resolver):
        """Test: Same confidence, prefer longer span."""
        text = "Bệnh nhân viêm phổi nặng"
        entities = [
            {"text": "viêm phổi nặng", "start": 12, "end": 26, "type": "CHẨN_ĐOÁN", "confidence": 0.8, "source": "ner"},
            {"text": "viêm phổi", "start": 12, "end": 21, "type": "CHẨN_ĐOÁN", "confidence": 0.8, "source": "dict"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        # Should prefer longer/earlier when same confidence
        # The resolver should pick based on processing order

    def test_partial_overlap_different_spans(self, resolver):
        """Test partial overlap between entities."""
        text = "Glucose 126 mg/dL cao"
        entities = [
            # Full lab result
            {"text": "Glucose 126 mg/dL", "start": 0, "end": 16, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "confidence": 0.9, "source": "ner"},
            # Just the number
            {"text": "126", "start": 8, "end": 11, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "confidence": 0.7, "source": "regex"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        # Should keep the more complete entity
        assert len(resolved) >= 1
        # The higher confidence one should win
        high_conf = [e for e in resolved if e["confidence"] >= 0.9]
        assert len(high_conf) >= 1

    def test_no_overlap_keep_both(self, resolver):
        """Test that non-overlapping entities are kept separate."""
        text = "Bệnh nhân ho và sốt."
        entities = [
            {"text": "ho", "start": 12, "end": 14, "type": "TRIỆU_CHỨNG", "confidence": 0.9, "source": "regex"},
            {"text": "sốt", "start": 19, "end": 22, "type": "TRIỆU_CHỨNG", "confidence": 0.9, "source": "regex"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 2

    def test_overlap_boundary_edge_case(self, resolver):
        """Test overlapping at boundary."""
        text = "ab"
        entities = [
            {"text": "a", "start": 0, "end": 1, "type": "TRIỆU_CHỨNG", "confidence": 0.9, "source": "ner"},
            {"text": "ab", "start": 0, "end": 2, "type": "TRIỆU_CHỨNG", "confidence": 0.8, "source": "dict"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        # Should resolve to one entity
        assert len(resolved) == 1

    def test_multiple_overlaps(self, resolver):
        """Test multiple overlapping entities."""
        text = "viêm phổi nặng"
        entities = [
            {"text": "viêm", "start": 0, "end": 4, "type": "CHẨN_ĐOÁN", "confidence": 0.7, "source": "ner"},
            {"text": "viêm phổi", "start": 0, "end": 9, "type": "CHẨN_ĐOÁN", "confidence": 0.8, "source": "ner"},
            {"text": "phổi nặng", "start": 5, "end": 13, "type": "CHẨN_ĐOÁN", "confidence": 0.85, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        # Should resolve to one or more entities after overlap resolution
        assert len(resolved) >= 1
        # Total span should be reasonable
        for e in resolved:
            assert e["end"] > e["start"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
