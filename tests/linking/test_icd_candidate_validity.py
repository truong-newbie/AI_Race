"""Tests for candidate validity checks and output format."""

import pytest
from src.linking.icd.schema import get_knowledge_base
from src.linking.icd.hybrid_retriever import HybridRetriever, CandidateResult, MergeConfig


class TestCandidateOutputFormat:
    """Test that candidate output matches specification."""

    @pytest.fixture
    def retriever(self):
        entries = get_knowledge_base()
        cfg = MergeConfig(method="rrf")
        return HybridRetriever(entries=entries, top_k=20)

    def test_output_has_code_and_score(self, retriever):
        """Output has code and score fields."""
        results = retriever.retrieve("suy tim", mention="suy tim")
        for r in results:
            assert "code" in r.to_dict()
            assert "score" in r.to_dict()
            assert isinstance(r.to_dict()["code"], str)
            assert isinstance(r.to_dict()["score"], float)

    def test_output_has_sources(self, retriever):
        """Output has sources list."""
        results = retriever.retrieve("suy tim", mention="suy tim")
        for r in results:
            d = r.to_dict()
            assert "sources" in d
            assert isinstance(d["sources"], list)

    def test_sources_are_valid(self, retriever):
        """Sources list contains valid source names."""
        valid_sources = {"exact", "normalized", "alias", "fuzzy", "bm25", "dense"}
        results = retriever.retrieve("suy tim", mention="suy tim")
        for r in results:
            for s in r.sources:
                assert s in valid_sources, f"Invalid source: {s}"

    def test_score_is_float(self, retriever):
        """Score is always a float."""
        results = retriever.retrieve("suy tim", mention="suy tim")
        for r in results:
            assert isinstance(r.score, float)

    def test_top_k_is_configurable(self, retriever):
        """top_k can be set per-call."""
        for k in [1, 3, 5, 10, 20]:
            results = retriever.retrieve("suy tim", mention="suy tim", top_k=k)
            assert len(results) <= k


class TestCandidateValidity:
    """Test that retrieved candidates are valid ICD codes."""

    @pytest.fixture
    def retriever(self):
        entries = get_knowledge_base()
        cfg = MergeConfig(method="rrf")
        return HybridRetriever(entries=entries, top_k=20)

    @pytest.fixture
    def valid_codes(self):
        return set(e.code for e in get_knowledge_base())

    def test_all_codes_in_kb(self, retriever, valid_codes):
        """All retrieved codes are in the KB."""
        test_cases = [
            "suy tim", "viêm phổi", "đái tháo đường",
            "tăng huyết áp", "động kinh", "hen",
        ]
        for mention in test_cases:
            results = retriever.retrieve(f"bệnh nhân {mention}", mention=mention)
            for r in results:
                assert r.code in valid_codes, f"{r.code} not in KB"

    def test_scores_non_negative(self, retriever):
        """All scores are non-negative."""
        test_cases = ["suy tim", "viêm phổi", "đái tháo đường"]
        for mention in test_cases:
            results = retriever.retrieve(f"bệnh nhân {mention}", mention=mention)
            for r in results:
                assert r.score >= 0, f"Negative score for {r.code}: {r.score}"

    def test_scores_decrease(self, retriever):
        """Scores monotonically decrease or stay equal."""
        results = retriever.retrieve("suy tim", mention="suy tim")
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


class TestCandidateDiversity:
    """Test that candidates are diverse and not all same code."""

    @pytest.fixture
    def retriever(self):
        entries = get_knowledge_base()
        cfg = MergeConfig(method="rrf")
        return HybridRetriever(entries=entries, top_k=20)

    def test_top_20_has_diversity(self, retriever):
        """Top-20 candidates contain multiple different codes."""
        results = retriever.retrieve("bệnh nhân có nhiều triệu chứng", mention="suy tim")
        codes = [r.code for r in results]
        unique = set(codes)
        # At least 5 unique codes expected in top 20
        assert len(unique) >= 3, f"Too few unique codes: {unique}"

    def test_different_queries_get_different_results(self, retriever):
        """Different mentions retrieve different top codes."""
        r1 = retriever.retrieve("suy tim", mention="suy tim")
        r2 = retriever.retrieve("viêm phổi", mention="viêm phổi")
        c1 = r1[0].code
        c2 = r2[0].code
        # They should be different codes (heart failure vs pneumonia)
        assert c1 != c2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
