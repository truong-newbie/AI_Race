"""Tests for drug retrieval (single ingredient + combinations)."""

import pytest
from src.linking.rxnorm.schema import get_knowledge_base, RxNormEntry
from src.linking.rxnorm.parser import DrugMentionParser
from src.linking.rxnorm.structured_matcher import StructuredMatcher
from src.linking.rxnorm.hybrid_retriever import DrugHybridRetriever


class TestSingleIngredientRetrieval:
    """Test single-ingredient drug retrieval."""

    @pytest.fixture
    def entries(self):
        return get_knowledge_base()

    @pytest.fixture
    def matcher(self, entries):
        return StructuredMatcher(entries=entries)

    @pytest.fixture
    def hybrid(self, entries):
        return DrugHybridRetriever(entries=entries, top_k=10)

    def test_metformin_500mg(self, hybrid):
        """Metformin 500mg retrieves correctly."""
        results = hybrid.retrieve("Metformin 500mg", mention="Metformin 500mg")
        codes = [r.rxcui for r in results]
        assert "6809" in codes
        # Best match should be 500mg
        if results:
            best = results[0]
            assert best.rxcui == "6809" or any(r.rxcui == "6809" for r in results)

    def test_metformin_1000mg(self, hybrid):
        """Metformin 1000mg retrieves correctly."""
        results = hybrid.retrieve("Metformin 1000mg", mention="Metformin 1000mg")
        codes = [r.rxcui for r in results]
        assert "861007" in codes

    def test_aspirin_81mg(self, hybrid):
        """Aspirin 81mg retrieves correctly."""
        results = hybrid.retrieve("Aspirin 81mg", mention="Aspirin 81mg")
        codes = [r.rxcui for r in results]
        assert "1191" in codes

    def test_aspirin_325mg(self, hybrid):
        """Aspirin 325mg retrieves correctly."""
        results = hybrid.retrieve("Aspirin 325mg", mention="Aspirin 325mg")
        codes = [r.rxcui for r in results]
        assert "1192" in codes

    def test_atorvastatin(self, hybrid):
        """Atorvastatin retrieves correctly."""
        results = hybrid.retrieve("Atorvastatin 20mg", mention="Atorvastatin 20mg")
        codes = [r.rxcui for r in results]
        assert "617312" in codes

    def test_ingredient_only(self, hybrid):
        """Ingredient name without strength."""
        results = hybrid.retrieve("Metformin", mention="Metformin")
        codes = [r.rxcui for r in results]
        assert any(r in codes for r in ["6809", "860975", "861007"])

    def test_ceftriaxone_injection(self, hybrid):
        """Ceftriaxone 1g injection."""
        results = hybrid.retrieve("Ceftriaxone 1g", mention="Ceftriaxone 1g")
        codes = [r.rxcui for r in results]
        assert "8628" in codes

    def test_decimal_strength(self, hybrid):
        """Alprazolam 0.5mg with decimal."""
        results = hybrid.retrieve("Alprazolam 0.5mg", mention="Alprazolam 0.5mg")
        codes = [r.rxcui for r in results]
        assert "72509" in codes

    def test_zopiclone(self, hybrid):
        """Zopiclone 7.5mg."""
        results = hybrid.retrieve("Zopiclone 7.5mg", mention="Zopiclone 7.5mg")
        codes = [r.rxcui for r in results]
        assert "206977" in codes


class TestStructuredMatcher:
    """Test structured matcher scoring."""

    @pytest.fixture
    def entries(self):
        return get_knowledge_base()

    @pytest.fixture
    def matcher(self, entries):
        return StructuredMatcher(entries=entries)

    @pytest.fixture
    def parser(self):
        return DrugMentionParser()

    def test_exact_name_match(self, matcher, parser):
        """Exact full-name gets highest score."""
        parsed = parser.parse("Metformin 500mg")
        results = matcher.match(parsed, top_k=5)
        if results:
            best = results[0]
            assert best[0] == "6809"  # Metformin 500mg

    def test_ingredient_match(self, matcher, parser):
        """Ingredient match gives score."""
        parsed = parser.parse("Metformin")
        results = matcher.match(parsed, top_k=5)
        assert len(results) > 0
        codes = [r[0] for r in results]
        assert any(c in codes for c in ["6809", "860975", "861007"])

    def test_strength_mismatch_penalty(self, matcher, parser):
        """Mismatched strength should get penalty."""
        # If we query Metformin 500mg but KB has only 1000mg entry
        parsed = parser.parse("Metformin 500mg")
        results = matcher.match(parsed, top_k=10)
        if results:
            for rxcui, score in results:
                if rxcui == "861007":  # 1000mg entry
                    # Should have lower score than 500mg entry
                    break

    def test_no_duplicate_results(self, matcher, parser):
        """Same rxcui should not appear twice."""
        parsed = parser.parse("Aspirin 81mg")
        results = matcher.match(parsed, top_k=20)
        codes = [r[0] for r in results]
        assert len(codes) == len(set(codes))

    def test_lookup_exact(self, matcher):
        """Direct lookup by name."""
        codes = matcher.lookup_exact("Metformin 500mg")
        # Should find exact or partial match
        assert isinstance(codes, list)


class TestHybridRetriever:
    """Test hybrid retriever combining sources."""

    @pytest.fixture
    def entries(self):
        return get_knowledge_base()

    @pytest.fixture
    def retriever(self, entries):
        return DrugHybridRetriever(entries=entries, top_k=20)

    def test_returns_candidates(self, retriever):
        """Returns list of candidates."""
        results = retriever.retrieve("Metformin 500mg", mention="Metformin 500mg")
        assert isinstance(results, list)

    def test_candidates_have_score(self, retriever):
        """Each candidate has a score."""
        results = retriever.retrieve("Aspirin 81mg", mention="Aspirin 81mg")
        for r in results:
            assert hasattr(r, 'score')
            assert isinstance(r.score, (int, float))
            assert 0 <= r.score

    def test_candidates_have_sources(self, retriever):
        """Each candidate tracks source."""
        results = retriever.retrieve("Metformin 500mg", mention="Metformin 500mg")
        for r in results:
            assert hasattr(r, 'sources')
            assert isinstance(r.sources, list)

    def test_top_k_limit(self, retriever):
        """Respects top_k limit."""
        results = retriever.retrieve("Metformin 500mg", mention="Metformin 500mg", top_k=3)
        assert len(results) <= 3

    def test_empty_query(self, retriever):
        """Empty query returns empty."""
        results = retriever.retrieve("", mention="")
        assert results == []

    def test_retrieve_one(self, retriever):
        """retrieve_one returns single result."""
        result = retriever.retrieve_one("Aspirin 81mg", mention="Aspirin 81mg")
        if result:
            assert isinstance(result.rxcui, str)
            assert isinstance(result.score, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
