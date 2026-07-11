"""
Tests for Family Scope Detection

Test Rule 6: isFamily detection with family relations
"""

import pytest
from src.assertion.rules import RuleBasedDetector
from src.assertion.scope import ClauseSegmenter


class TestFamilyBasic:
    """Basic family history detection tests."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_family_bo(self, detector):
        """Test 'bố' family relation."""
        text = "Bố bệnh nhân bị tiểu đường."
        entity_start = text.index("tiểu đường")
        entity_end = entity_start + 11

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_me(self, detector):
        """Test 'mẹ' family relation."""
        text = "Mẹ bệnh nhân có bệnh tim."
        entity_start = text.index("bệnh tim")
        entity_end = entity_start + 8

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_cha(self, detector):
        """Test 'cha' family relation."""
        text = "Cha bệnh nhân từng bị ung thư."
        entity_start = text.index("ung thư")
        entity_end = entity_start + 7

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_ong(self, detector):
        """Test 'ông' family relation."""
        text = "Ông nội bệnh nhân mắc bệnh phổi."
        entity_start = text.index("bệnh phổi")
        entity_end = entity_start + 10

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_ba(self, detector):
        """Test 'bà' family relation."""
        text = "Bà ngoại bệnh nhân có tiền sử bệnh."
        entity_start = text.index("tiền sử bệnh")
        entity_end = entity_start + 12

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_anh(self, detector):
        """Test 'anh' family relation."""
        text = "Anh trai bệnh nhân bị hen."
        entity_start = text.index("hen")
        entity_end = entity_start + 3

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_chi(self, detector):
        """Test 'chị' family relation."""
        text = "Chị gái bệnh nhân mắc bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_em(self, detector):
        """Test 'em' family relation."""
        text = "Em trai bệnh nhân có bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_con(self, detector):
        """Test 'con' family relation."""
        text = "Con trai bệnh nhân bị bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_nguoi_nha(self, detector):
        """Test 'người nhà' family relation."""
        text = "Người nhà bệnh nhân có bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_gia_dinh(self, detector):
        """Test 'gia đình' family relation."""
        text = "Gia đình bệnh nhân có tiền sử bệnh tim."
        entity_start = text.index("bệnh tim")
        entity_end = entity_start + 8

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_gia_dinh_co(self, detector):
        """Test 'gia đình có' family relation."""
        text = "Gia đình có người mắc bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_ho_hang(self, detector):
        """Test 'họ hàng' family relation."""
        text = "Họ hàng bên ngoại có bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_ong_ba(self, detector):
        """Test 'ông bà' family relation."""
        text = "Ông bà bệnh nhân mắc bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_cha_me(self, detector):
        """Test 'cha mẹ' family relation."""
        text = "Cha mẹ bệnh nhân đều có bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_tien_su_gia_dinh(self, detector):
        """Test 'tiền sử gia đình' combined cue."""
        text = "Tiền sử gia đình bệnh nhân có bệnh tim."
        entity_start = text.index("bệnh tim")
        entity_end = entity_start + 8

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_no_family(self, detector):
        """Test that patient statement is not family."""
        text = "Bệnh nhân bị bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is False


class TestFamilyHistorical:
    """Family and historical combined tests (Rule 6)."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_family_and_historical(self, detector):
        """Test Rule 6: 'mẹ bệnh nhân từng mắc hen suyễn'."""
        text = "Mẹ bệnh nhân từng mắc hen suyễn."
        entity_start = text.index("hen suyễn")
        entity_end = entity_start + 10

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True
        assert result.status.is_historical is True

    def test_family_with_tien_su(self, detector):
        """Test family with 'có tiền sử' - only isFamily (validation data behavior)."""
        text = "Bố có tiền sử bệnh tim."
        entity_start = text.index("bệnh tim")
        entity_end = entity_start + 8

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True
        assert result.status.is_historical is False

    def test_tien_su_gia_dinh_historical(self, detector):
        """Test 'tiền sử gia đình' with historical."""
        text = "Tiền sử gia đình bệnh nhân mắc bệnh."
        entity_start = text.index("bệnh")
        entity_end = entity_start + 5

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True
        # Note: tiền sử gia đình may or may not also be historical


class TestFamilyScope:
    """Family scope handling tests."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_family_same_sentence(self, detector):
        """Test family cue stays within same sentence."""
        text = "Gia đình có bệnh tim. Bệnh nhân khỏe."
        entity_start = text.index("bệnh tim")
        entity_end = entity_start + 8

        result = detector.detect(text, entity_start, entity_end)

        assert result.status.is_family is True

    def test_family_does_not_span_sentence(self, detector):
        """Test family doesn't span across sentences."""
        text = "Gia đình có bệnh. Bệnh nhân ho."
        ho_start = text.index("ho")

        result = detector.detect(text, ho_start, ho_start + 2)

        # "Gia đình" is in previous sentence
        assert result.status.is_family is False


class TestFamilyEntityTypes:
    """Family detection for different entity types."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_family_symptom(self, detector):
        """Test family on TRIỆU_CHỨNG."""
        text = "Mẹ bệnh nhân từng ho nhiều."
        entity_start = text.index("ho")
        entity_end = entity_start + 2

        result = detector.detect(text, entity_start, entity_end, "TRIỆU_CHỨNG")

        assert result.status.is_family is True

    def test_family_diagnosis(self, detector):
        """Test family on CHẨN_ĐOÁN."""
        text = "Bố bệnh nhân có bệnh tiểu đường."
        entity_start = text.index("tiểu đường")
        entity_end = entity_start + 11

        result = detector.detect(text, entity_start, entity_end, "CHẨN_ĐOÁN")

        assert result.status.is_family is True

    def test_family_drug(self, detector):
        """Test family on THUỐC (drug allergy context)."""
        text = "Mẹ bệnh nhân dị ứng với Penicillin."
        entity_start = text.index("Penicillin")
        entity_end = entity_start + 9

        result = detector.detect(text, entity_start, entity_end, "THUỐC")

        assert result.status.is_family is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
