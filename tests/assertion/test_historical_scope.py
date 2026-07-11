"""
Tests for Historical Scope Detection

Test Rule 7: isHistorical detection
"""

import pytest
from src.assertion.rules import RuleBasedDetector
from src.assertion.scope import ClauseSegmenter


class TestHistoricalBasic:
    """Basic historical detection tests."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_historical_tien_su(self, detector):
        """Test 'tiền sử' historical cue."""
        text = "Có tiền sử hen suyễn."
        entity_start = text.index("hen suyễn")
        entity_end = entity_start + 10

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_historical_tien_su_benh(self, detector):
        """Test 'tiền sử bệnh' historical cue."""
        text = "Tiền sử bệnh tiểu đường."
        entity_start = text.index("tiểu đường")
        entity_end = entity_start + 11

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_historical_da_tung(self, detector):
        """Test 'đã từng' historical cue."""
        text = "Bệnh nhân đã từng bị viêm phổi."
        entity_start = text.index("viêm phổi")
        entity_end = entity_start + 10

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_historical_tung(self, detector):
        """Test 'từng' historical cue."""
        text = "Từng điều trị tại bệnh viện."
        entity_start = text.index("điều trị")
        entity_end = entity_start + 9

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_historical_truoc_day(self, detector):
        """Test 'trước đây' historical cue."""
        text = "Trước đây từng mắc bệnh."
        entity_start = text.index("mắc bệnh")
        entity_end = entity_start + 9

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_historical_truoc_do(self, detector):
        """Test 'trước đó' historical cue."""
        text = "Trước đó đã phẫu thuật."
        entity_start = text.index("phẫu thuật")
        entity_end = entity_start + 11

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_historical_cach_day(self, detector):
        """Test 'cách đây' historical cue."""
        text = "Cách đây 2 năm mắc bệnh."
        entity_start = text.index("mắc bệnh")
        entity_end = entity_start + 9

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_historical_tung_dieu_tri(self, detector):
        """Test 'từng điều trị' historical cue."""
        text = "Từng điều trị bằng thuốc."
        entity_start = text.index("thuốc")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_historical_da_su_dung(self, detector):
        """Test 'đã sử dụng' historical cue."""
        text = "Đã sử dụng thuốc hạ áp."
        entity_start = text.index("thuốc hạ áp")
        entity_end = entity_start + 11

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_historical_benh_cu(self, detector):
        """Test 'bệnh cũ' historical cue."""
        text = "Bệnh cũ tái phát."
        entity_start = text.index("tái phát")
        entity_end = entity_start + 9

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_no_historical(self, detector):
        """Test that current statement is not historical."""
        text = "Bệnh nhân bị ho."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is False


class TestHistoricalScope:
    """Historical scope handling tests."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_historical_same_sentence(self, detector):
        """Test historical cue stays within same sentence."""
        text = "Tiền sử bệnh tim. Hiện tại khỏe."
        entity_start = text.index("bệnh tim")
        entity_end = entity_start + 8

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True

    def test_historical_does_not_span_sentence(self, detector):
        """Test historical doesn't span across sentences."""
        text = "Tiền sử bệnh. Hiện tại ho."
        ho_start = text.index("ho")

        result = detector.detect(text, ho_start, ho_start + 2)

        # "Tiền sử" is in previous sentence
        assert result.status.is_historical is False


class TestHistoricalEntityTypes:
    """Historical detection for different entity types."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_historical_symptom(self, detector):
        """Test historical on TRIỆU_CHỨNG."""
        text = "Tiền sử ho kéo dài."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end, "TRIỆU_CHỨNG")

        assert result.status.is_historical is True

    def test_historical_diagnosis(self, detector):
        """Test historical on CHẨN_ĐOÁN."""
        text = "Tiền sử bệnh tiểu đường."
        entity_start = text.index("tiểu đường")
        entity_end = entity_start + 11

        result = detector.detect(text, entity_start, entity_end, "CHẨN_ĐOÁN")

        assert result.status.is_historical is True

    def test_historical_drug(self, detector):
        """Test historical on THUỐC."""
        text = "Đã sử dụng Paracetamol trước đây."
        entity_start = text.index("Paracetamol")
        entity_end = entity_start + 11

        result = detector.detect(text, entity_start, entity_end, "THUỐC")

        assert result.status.is_historical is True


class TestHistoricalConfidence:
    """Historical confidence scoring tests."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_high_priority_cue(self, detector):
        """Test high priority cues get higher confidence."""
        text = "Có tiền sử bệnh tiểu đường."
        entity_start = text.index("tiểu đường")
        entity_end = entity_start + 11

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True
        assert result.status.confidence > 0.85

    def test_low_priority_cue(self, detector):
        """Test lower priority cues get normal confidence."""
        text = "Từng bị bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_historical is True
        assert result.status.confidence > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
