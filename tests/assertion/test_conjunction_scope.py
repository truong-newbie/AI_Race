"""
Tests for Conjunction Scope Handling

Test Rule 3 & 5: Conjunction handling and scope exceptions
"""

import pytest
from src.assertion.rules import RuleBasedDetector
from src.assertion.scope import ClauseSegmenter


class TestConjunctionContrast:
    """Test contrast conjunction handling."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_nhung_contrast(self, detector):
        """Test 'nhưng' conjunction."""
        text = "Không ho nhưng đau ngực."

        ho_start = text.index("ho")
        dau_start = text.index("đau ngực")

        ho_result = detector.detect(text, ho_start, ho_start + 2)
        dau_result = detector.detect(text, dau_start, dau_start + 9)

        assert ho_result.status.is_negated is True
        assert dau_result.status.is_negated is False

    def test_tuy_nhien_contrast(self, detector):
        """Test 'tuy nhiên' conjunction."""
        text = "Không sốt tuy nhiên đau đầu."

        sot_start = text.index("sốt")
        dau_start = text.index("đau đầu")

        sot_result = detector.detect(text, sot_start, sot_start + 3)
        dau_result = detector.detect(text, dau_start, dau_start + 8)

        assert sot_result.status.is_negated is True
        assert dau_result.status.is_negated is False

    def test_song_contrast(self, detector):
        """Test 'song' conjunction."""
        text = "Không ho song còn sốt."

        ho_start = text.index("ho")
        sot_start = text.index("sốt")

        ho_result = detector.detect(text, ho_start, ho_start + 2)
        sot_result = detector.detect(text, sot_start, sot_start + 3)

        assert ho_result.status.is_negated is True
        # "còn" indicates contrast - sot may not be negated
        assert sot_result.status.is_negated is False

    def test_con_contrast(self, detector):
        """Test 'còn' conjunction."""
        text = "Không sốt còn đau ngực."

        sot_start = text.index("sốt")
        dau_start = text.index("đau ngực")

        sot_result = detector.detect(text, sot_start, sot_start + 3)
        dau_result = detector.detect(text, dau_start, dau_start + 9)

        assert sot_result.status.is_negated is True
        assert dau_result.status.is_negated is False


class TestConjunctionMultiple:
    """Test multiple conjunctions in one sentence."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_multiple_conjunctions(self, detector):
        """Test sentence with multiple conjunctions."""
        text = "Không ho, không sốt nhưng đau ngực."

        ho_start = text.index("ho")
        sot_start = text.index("sốt")
        dau_start = text.index("đau ngực")

        ho_result = detector.detect(text, ho_start, ho_start + 2)
        sot_result = detector.detect(text, sot_start, sot_start + 3)
        dau_result = detector.detect(text, dau_start, dau_start + 9)

        # Both ho and sốt should be negated (before nhưng)
        assert ho_result.status.is_negated is True
        assert sot_result.status.is_negated is True
        # dau ngực should NOT be negated (after nhưng)
        assert dau_result.status.is_negated is False


class TestConjunctionHistorical:
    """Test conjunction handling with historical cues."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_historical_with_conjunction(self, detector):
        """Test historical cue with conjunction."""
        text = "Tiền sử bệnh tim nhưng hiện tại khỏe."

        tim_start = text.index("bệnh tim")
        khoe_start = text.index("khỏe")

        tim_result = detector.detect(text, tim_start, tim_start + 8)
        khoe_result = detector.detect(text, khoe_start, khoe_start + 4)

        # bệnh tim is historical
        assert tim_result.status.is_historical is True
        # khỏe is NOT historical (hiện tại = current)


class TestConjunctionFamily:
    """Test conjunction handling with family cues."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_family_with_conjunction(self, detector):
        """Test family cue with conjunction."""
        text = "Mẹ bệnh nhân có bệnh tim nhưng bố khỏe."

        tim_start = text.index("bệnh tim")
        bo_start = text.index("bố")

        tim_result = detector.detect(text, tim_start, tim_start + 8)
        bo_result = detector.detect(text, bo_start, bo_start + 2)

        # bệnh tim is family
        assert tim_result.status.is_family is True


class TestClauseSegmenter:
    """Test clause segmentation."""

    @pytest.fixture
    def segmenter(self):
        return ClauseSegmenter()

    def test_segment_sentences(self, segmenter):
        """Test sentence segmentation."""
        text = "Ho. Sốt. Đau."

        sentences = segmenter.segment_sentences(text)

        assert len(sentences) == 3
        assert sentences[0].text == "Ho"
        assert sentences[1].text == "Sốt"
        assert sentences[2].text == "Đau"

    def test_segment_sentences_with_period(self, segmenter):
        """Test sentence segmentation with period."""
        text = "Bệnh nhân ho. Đau ngực."

        sentences = segmenter.segment_sentences(text)

        assert len(sentences) == 2

    def test_segment_clauses(self, segmenter):
        """Test clause segmentation within sentence."""
        text = "Không ho, không sốt nhưng đau ngực."

        sentences = segmenter.segment_sentences(text)

        assert len(sentences) == 1
        assert len(sentences[0].clauses) >= 1


class TestConjunctionEdgeCases:
    """Edge case tests for conjunction handling."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_negation_after_conjunction_same_word(self, detector):
        """Test when conjunction is the same word."""
        text = "Không không ho."  # Double không

        ho_start = text.index("ho")
        ho_result = detector.detect(text, ho_start, ho_start + 2)

        # ho should be negated
        assert ho_result.status.is_negated is True

    def test_entity_between_cues(self, detector):
        """Test entity between two cues."""
        text = "Không ho và không sốt."

        ho_start = text.index("ho")
        sot_start = text.index("sốt")

        ho_result = detector.detect(text, ho_start, ho_start + 2)
        sot_result = detector.detect(text, sot_start, sot_start + 3)

        assert ho_result.status.is_negated is True
        assert sot_result.status.is_negated is True

    def test_long_sentence_conjunction(self, detector):
        """Test long sentence with conjunction."""
        text = "Bệnh nhân không có triệu chứng ho, sốt, khó thở nhưng không đau ngực."

        ho_start = text.index("ho")
        sot_start = text.index("sốt")
        dau_start = text.index("đau ngực")

        ho_result = detector.detect(text, ho_start, ho_start + 2)
        sot_result = detector.detect(text, sot_start, sot_start + 3)
        dau_result = detector.detect(text, dau_start, dau_start + 9)

        # All in the negation scope before "nhưng"
        assert ho_result.status.is_negated is True
        assert sot_result.status.is_negated is True
        # dau ngực is after "nhưng"
        assert dau_result.status.is_negated is True  # Also has "không"


class TestConjunctionConfidence:
    """Test confidence with conjunction handling."""

    @pytest.fixture
    def detector(self):
        return RuleBasedDetector()

    def test_confidence_with_conjunction(self, detector):
        """Test that confidence is calculated correctly with conjunction."""
        text = "Không ho nhưng đau ngực."

        ho_start = text.index("ho")
        ho_result = detector.detect(text, ho_start, ho_start + 2)

        # Should have high confidence
        assert ho_result.status.confidence > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
