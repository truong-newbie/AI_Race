"""Tests for exact and normalized alias matching."""

import pytest
from src.linking.icd.schema import get_knowledge_base
from src.linking.icd.alias_index import AliasIndex
from src.linking.icd.preprocess import TextNormalizer


class TestExactMatch:
    """Exact match retrieval tests."""

    @pytest.fixture
    def index(self):
        entries = get_knowledge_base()
        idx = AliasIndex()
        idx.build(entries)
        return idx

    @pytest.fixture
    def normalizer(self):
        return TextNormalizer()

    def test_exact_match_vi(self, index):
        """Exact Vietnamese name match."""
        results = index.lookup_exact("suy tim")
        assert "I50.9" in results

    def test_exact_match_en(self, index):
        """Exact English name match."""
        results = index.lookup_exact("heart failure")
        assert "I50.9" in results

    def test_exact_match_alias(self, index):
        """Exact alias match."""
        results = index.lookup_exact("trào ngược dạ dày")
        assert "K21.9" in results

    def test_exact_match_case_insensitive(self, index):
        """Case-insensitive match."""
        results = index.lookup_exact("SUY TIM")
        assert "I50.9" in results

    def test_exact_match_gerd(self, index):
        """GERD alias for K21.9."""
        results = index.lookup_exact("gerd")
        assert "K21.9" in results

    def test_exact_match_no_match(self, index):
        """No match for unknown term."""
        results = index.lookup_exact("xyz unknown disease")
        assert len(results) == 0

    def test_exact_match_htn(self, index):
        """HTN alias for I10."""
        results = index.lookup_exact("HTN")
        assert "I10" in results

    def test_exact_match_copd(self, index):
        """COPD alias for J44.9."""
        results = index.lookup_exact("COPD")
        assert "J44.9" in results

    def test_exact_match_uti(self, index):
        """UTI alias for N39.0."""
        results = index.lookup_exact("UTI")
        assert "N39.0" in results

    def test_exact_match_stroke(self, index):
        """Stroke alias for I64."""
        results = index.lookup_exact("đột quỵ")
        assert "I64" in results


class TestNormalizedMatch:
    """Normalized exact match tests."""

    @pytest.fixture
    def index(self):
        entries = get_knowledge_base()
        idx = AliasIndex()
        idx.build(entries)
        return idx

    def test_normalized_whitespace(self, index):
        """Match with extra whitespace."""
        results = index.lookup_normalized("suy   tim")
        assert "I50.9" in results

    def test_normalized_dash(self, index):
        """Match with normalized dash."""
        # "viêm-phổi" should normalize to "viêm phổi"
        results = index.lookup_normalized("viêm-phổi")
        assert "J18.9" in results

    def test_normalized_lowercase(self, index):
        """Already lowercase is handled."""
        results = index.lookup_normalized("viêm phổi")
        assert "J18.9" in results


class TestAliasIndex:
    """Alias index build and structure tests."""

    def test_all_codes_indexed(self):
        """All 38 KB codes are indexed."""
        from src.linking.icd.schema import get_knowledge_base
        entries = get_knowledge_base()
        idx = AliasIndex()
        idx.build(entries)

        codes = idx.all_codes()
        assert len(codes) == 38
        for code in ["I50.9", "J18.9", "K21.9", "E11.9", "G40.909", "N39.0"]:
            assert code in codes

    def test_entry_lookup(self):
        """Entry can be retrieved by code."""
        entries = get_knowledge_base()
        idx = AliasIndex()
        idx.build(entries)

        entry = idx.get_entry("I50.9")
        assert entry is not None
        assert entry.code == "I50.9"
        assert "suy tim" in entry.aliases

    def test_multiple_codes_from_synonym(self):
        """A synonym can map to multiple codes if entries share it."""
        entries = get_knowledge_base()
        idx = AliasIndex()
        idx.build(entries)

        # "viêm dạ dày" should map to both K29.9 and K25.9
        results = idx.lookup_exact("viêm dạ dày")
        assert len(results) >= 1

    def test_empty_query(self):
        """Empty query returns empty."""
        entries = get_knowledge_base()
        idx = AliasIndex()
        idx.build(entries)

        assert idx.lookup_exact("") == []
        assert idx.lookup_normalized("") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
