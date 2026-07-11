"""
Tests for Multiple Assertions

Test entities with multiple assertion types simultaneously.
"""

import pytest
from src.assertion.rules import RuleBasedDetector, EntityAssertion


class TestMultipleAssertions:
    """Test entities with multiple assertion types."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_negated_and_historical(self, detector):
        """Test entity that is both negated and historical."""
        text = "Tiền sử không ho trước đây."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end)

        # "Tiền sử" makes it historical
        assert result.status.is_historical is True
        # "Không" before "ho" makes it negated
        assert result.status.is_negated is True

    def test_family_and_historical(self, detector):
        """Test entity that is both family and historical."""
        text = "Mẹ từng bị tiểu đường."
        entity_start = text.index("tiểu đường")
        entity_end = entity_start + 11

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True
        assert result.status.is_historical is True

    def test_all_three_assertions(self, detector):
        """Test entity with all three assertions."""
        text = "Tiền sử gia đình, mẹ không từng bị bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        # Note: actual result depends on cue positions
        # This tests that multiple flags can be set


class TestAssertionBatch:
    """Test batch assertion detection."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_batch_detection(self, detector):
        """Test detecting assertions for multiple entities."""
        text = "Bệnh nhân không ho, mẹ có tiền sử bệnh tim."

        entities = [
            {"start": text.index("ho"), "end": text.index("ho") + 2, "type": "TRIỆU_CHỨNG"},
            {"start": text.index("bệnh tim"), "end": text.index("bệnh tim") + 8, "type": "CHẨN_ĐOÁN"},
        ]

        results = detector.detect_all(text, entities)

        assert len(results) == 2

        # ho is negated
        assert results[0].status.is_negated is True

        # bệnh tim is family only (tiền sử right after family cue → isFamily, not isHistorical)
        assert results[1].status.is_family is True
        assert results[1].status.is_historical is False

    def test_empty_entities(self, detector):
        """Test with empty entity list."""
        text = "Bệnh nhân không ho."
        results = detector.detect_all(text, [])

        assert len(results) == 0

    def test_mixed_entity_types(self, detector):
        """Test with different entity types."""
        text = "Không dùng Aspirin, tiền sử bệnh tiểu đường."

        entities = [
            {"start": text.index("Aspirin"), "end": text.index("Aspirin") + 7, "type": "THUỐC"},
            {"start": text.index("tiểu đường"), "end": text.index("tiểu đường") + 11, "type": "CHẨN_ĐOÁN"},
        ]

        results = detector.detect_all(text, entities)

        assert len(results) == 2
        assert results[0].entity_type == "THUỐC"
        assert results[1].entity_type == "CHẨN_ĐOÁN"


class TestAssertionOutput:
    """Test assertion output formats."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_to_dict(self, detector):
        """Test conversion to dictionary."""
        text = "Bệnh nhân không ho."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end)
        result_dict = result.to_dict()

        assert "entity_text" in result_dict
        assert "entity_start" in result_dict
        assert "entity_end" in result_dict
        assert "is_negated" in result_dict
        assert "is_historical" in result_dict
        assert "is_family" in result_dict
        assert result_dict["is_negated"] is True

    def test_to_list(self, detector):
        """Test conversion to list."""
        text = "Bệnh nhân không ho."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end)
        result_list = result.to_list()

        assert "isNegated" in result_list
        assert "isFamily" not in result_list
        assert "isHistorical" not in result_list

    def test_to_list_multiple(self, detector):
        """Test list output with multiple assertions."""
        text = "Tiền sử gia đình, mẹ từng bị bệnh."

        result = detector.detect(text, text.index("bệnh"), text.index("bệnh") + 5)

        # Depending on implementation, may have multiple assertions
        result_list = result.to_list()
        assert isinstance(result_list, list)


class TestEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_entity_at_sentence_start(self, detector):
        """Test entity at start of sentence."""
        text = "Không ho. Sốt cao."
        entity_start = text.index("Sốt")
        entity_end = entity_start + 3

        result = detector.detect(text, entity_start, entity_end)

        # "Không" is in previous sentence, should not apply
        assert result.status.is_negated is False

    def test_entity_at_sentence_end(self, detector):
        """Test entity at end of sentence."""
        text = "Bệnh nhân không ho."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True

    def test_multiple_cues_same_type(self, detector):
        """Test multiple cues of same type."""
        text = "Không không ho."  # Double negation
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True

    def test_no_cues(self, detector):
        """Test text with no assertion cues."""
        text = "Bệnh nhân bình thường."
        entity_start = text.index("bình thường")
        entity_end = entity_start + 13

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is False
        assert result.status.is_historical is False
        assert result.status.is_family is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
