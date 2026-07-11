"""
Tests for Negation Scope Detection

Test Rule 1-5: isNegated detection with scope handling
"""

import pytest
from src.assertion.rules import RuleBasedDetector, EntityAssertion
from src.assertion.scope import ClauseSegmenter, find_cue_matches
from src.assertion.cues import CueType


class TestNegationBasic:
    """Basic negation detection tests."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_simple_negation_khong(self, detector):
        """Test simple negation with 'không'."""
        text = "Bệnh nhân không ho."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True
        assert "không" in result.status.cues_used

    def test_simple_negation_chua(self, detector):
        """Test simple negation with 'chưa'."""
        text = "Bệnh nhân chưa sốt."
        entity_start = text.index("sốt")
        entity_end = entity_start + 3

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True

    def test_negation_khong_co(self, detector):
        """Test 'không có' negation."""
        text = "Không có dấu hiệu ho."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True

    def test_negation_chua_ghi_nhan(self, detector):
        """Test 'chưa ghi nhận' negation."""
        text = "Chưa ghi nhận bất thường."
        entity_start = text.index("bất thường")
        entity_end = entity_start + 11

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True

    def test_negation_khong_ghi_nhan(self, detector):
        """Test 'không ghi nhận' negation."""
        text = "Không ghi nhận bệnh phổi."
        entity_start = text.index("bệnh phổi")
        entity_end = entity_start + 10

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True

    def test_negation_khong_thay(self, detector):
        """Test 'không thấy' negation."""
        text = "Không thấy tổn thương."
        entity_start = text.index("tổn thương")
        entity_end = entity_start + 10

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True

    def test_negation_loai_tru(self, detector):
        """Test 'loại trừ' negation."""
        text = "Loại trừ bệnh lao."
        entity_start = text.index("bệnh lao")
        entity_end = entity_start + 8

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True

    def test_negation_am_tinh(self, detector):
        """Test 'âm tính' negation."""
        text = "Xét nghiệm âm tính với HIV."
        entity_start = text.index("HIV")
        entity_end = entity_start + 3

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True

    def test_no_negation(self, detector):
        """Test that normal statement is not negated."""
        text = "Bệnh nhân ho đờm."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is False


class TestNegationScope:
    """Negation scope handling tests."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    @pytest.fixture
    def segmenter(self):
        return ClauseSegmenter()

    def test_negation_same_sentence(self, detector):
        """Test negation stays within same sentence."""
        text = "Bệnh nhân không ho. Đau ngực."
        ho_start = text.index("ho")
        ho_end = ho_start + 2

        result = detector.detect(text, ho_start, ho_end)

        assert result.status.is_negated is True

    def test_negation_does_not_span_sentence(self, detector):
        """Test negation doesn't span across sentences."""
        text = "Không ho. Bệnh nhân sốt."
        sot_start = text.index("sốt")
        sot_end = sot_start + 3

        result = detector.detect(text, sot_start, sot_end)

        # "Không" is in previous sentence, should not apply
        assert result.status.is_negated is False

    def test_negation_entity_list(self, detector):
        """Test negation applies to entity list."""
        text = "Không ho, sốt, khó thở."

        ho_start = text.index("ho")
        sot_start = text.index("sốt")
        kho_start = text.index("khó thở")

        ho_result = detector.detect(text, ho_start, ho_start + 2)
        sot_result = detector.detect(text, sot_start, sot_start + 3)
        kho_result = detector.detect(text, kho_start, kho_start + 8)

        assert ho_result.status.is_negated is True
        assert sot_result.status.is_negated is True
        assert kho_result.status.is_negated is True

    def test_negation_conjunction_exception(self, detector):
        """Test Rule 5: 'không ho nhưng đau ngực' - only ho is negated."""
        text = "Không ho nhưng đau ngực."

        ho_start = text.index("ho")
        dau_start = text.index("đau ngực")

        ho_result = detector.detect(text, ho_start, ho_start + 2)
        dau_result = detector.detect(text, dau_start, dau_start + 9)

        # ho should be negated
        assert ho_result.status.is_negated is True
        # dau ngực should NOT be negated (it's after "nhưng")
        assert dau_result.status.is_negated is False

    def test_negation_after_conjunction(self, detector):
        """Test entities after conjunction are not negated."""
        text = "Không sốt tuy nhiên đau đầu."

        sot_start = text.index("sốt")
        dau_start = text.index("đau đầu")

        sot_result = detector.detect(text, sot_start, sot_start + 3)
        dau_result = detector.detect(text, dau_start, dau_start + 8)

        # sot should be negated
        assert sot_result.status.is_negated is True
        # dau dau should NOT be negated
        assert dau_result.status.is_negated is False


class TestNegationConfidence:
    """Negation confidence scoring tests."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_high_priority_cue_high_confidence(self, detector):
        """Test that specific patterns get higher confidence."""
        text = "Loại trừ bệnh lao phổi."
        entity_start = text.index("bệnh lao")
        entity_end = entity_start + 8

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True
        assert result.status.confidence > 0.8

    def test_low_priority_cue_normal_confidence(self, detector):
        """Test that general patterns get normal confidence."""
        text = "Không ho."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_negated is True
        assert result.status.confidence > 0.7


class TestNegationEntityTypes:
    """Negation for different entity types."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_negate_symptom(self, detector):
        """Test negation on TRIỆU_CHỨNG."""
        text = "Bệnh nhân không ho."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end, "TRIỆU_CHỨNG")

        assert result.entity_type == "TRIỆU_CHỨNG"
        assert result.status.is_negated is True

    def test_negate_diagnosis(self, detector):
        """Test negation on CHẨN_ĐOÁN."""
        text = "Loại trừ viêm phổi."
        entity_start = text.index("viêm phổi")
        entity_end = entity_start + 10

        result = detector.detect(text, entity_start, entity_end, "CHẨN_ĐOÁN")

        assert result.entity_type == "CHẨN_ĐOÁN"
        assert result.status.is_negated is True

    def test_negate_drug(self, detector):
        """Test negation on THUỐC."""
        text = "Không dùng Aspirin."
        entity_start = text.index("Aspirin")
        entity_end = entity_start + 7

        result = detector.detect(text, entity_start, entity_end, "THUỐC")

        assert result.entity_type == "THUỐC"
        assert result.status.is_negated is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
