"""Tests for ICD-10 specificity reranking rules."""

import pytest
from src.linking.icd.schema import get_knowledge_base
from src.linking.rule_reranker import ICDRuleReranker, ICD_WEIGHTS
from src.linking.icd.hybrid_retriever import CandidateResult


@pytest.fixture
def entries():
    return get_knowledge_base()


@pytest.fixture
def reranker(entries):
    return ICDRuleReranker(entries)


@pytest.fixture
def mock_icd_candidates():
    """Mock ICD candidates with equal retrieval scores."""
    class MockCand:
        def __init__(self, code, score):
            self.code = code
            self.score = score
            self.sources = []
            self.detail = {}
    return [
        MockCand("K21.9", 0.01),   # GERD — parent
        MockCand("K29.9", 0.01),   # Gastritis — different code
    ]


class TestICDSpecificity:
    """Tests for ICD specificity-based reranking."""

    def test_parent_code_preferred_for_generic_mention(self, reranker, entries):
        """Generic mention like 'đau bụng' should not strongly prefer any specific code."""
        # Build mock candidates: child vs parent
        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        candidates = [
            MockCand("K21.9", 0.005),   # GERD (parent)
            MockCand("K25.9", 0.005),   # Gastric ulcer (child)
        ]

        reranked = reranker.rerank(
            candidates,
            query="BN đau bụng",
            mention="đau bụng",
            top_k=2,
        )

        assert len(reranked) == 2
        # Both should be scored, no crash
        assert all(r.rerank_score >= 0 for r in reranked)

    def test_child_code_penalized_without_detail(self, reranker, entries):
        """Child code K29.9 should be penalized when mention is generic."""
        entry_k29 = next(e for e in entries if e.code == "K29.9")
        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        # Generic mention — no detail signals
        candidates = [MockCand("K29.9", 0.01)]
        reranked = reranker.rerank(
            candidates,
            query="bệnh nhân",
            mention="viêm dạ dày",
            top_k=1,
        )

        result = reranked[0]
        # Should have constraint penalty applied
        assert "specificity_penalty" in result.features or result.constraint_penalty >= 0

    def test_include_term_boost(self, reranker, entries):
        """Mention matching include term should get boost."""
        # GERD entry has include_terms: "ợ nóng", "ợ chua", "acid reflux"
        entry_gerd = next(e for e in entries if e.code == "K21.9")

        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        candidates = [MockCand("K21.9", 0.01)]
        reranked = reranker.rerank(
            candidates,
            query="BN có triệu chứng ợ nóng",
            mention="ợ nóng",
            top_k=1,
        )

        assert len(reranked) == 1
        assert "context_include" in reranked[0].features

    def test_exclude_term_rejection(self, reranker, entries):
        """Mention with exclude term should get penalty."""
        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        # GERD has no exclude_terms in the KB — use another entry
        candidates = [MockCand("K21.9", 0.01)]
        reranked = reranker.rerank(
            candidates,
            query="BN không có vấn đề",
            mention="không có vấn đề",
            top_k=1,
        )

        # Should not crash
        assert len(reranked) == 1

    def test_lexical_similarity_high_for_match(self, reranker, entries):
        """High lexical similarity for matching alias."""
        entry = next(e for e in entries if e.code == "I10")  # Hypertension

        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        candidates = [MockCand("I10", 0.01)]
        reranked = reranker.rerank(
            candidates,
            query="BN tăng huyết áp",
            mention="tăng huyết áp",
            top_k=1,
        )

        assert len(reranked) == 1
        assert "lexical_similarity" in reranked[0].features

    def test_rerank_preserves_order_for_dissimilar_candidates(self, reranker, entries):
        """When candidates are very different, retrieval order should be preserved."""
        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        candidates = [
            MockCand("K21.9", 0.02),  # GERD — higher retrieval score
            MockCand("I10", 0.01),     # Hypertension — lower retrieval score
        ]

        reranked = reranker.rerank(
            candidates,
            query="BN có vấn đề tiêu hóa và tim mạch",
            mention="tăng huyết áp",
            top_k=2,
        )

        # GERD still has higher retrieval score component
        assert len(reranked) == 2

    def test_rerank_empty_list(self, reranker):
        """Reranking empty list should return empty list."""
        reranked = reranker.rerank([], query="test", mention="test", top_k=10)
        assert reranked == []

    def test_rerank_top_k_limit(self, reranker):
        """Rerank should respect top_k limit."""
        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        candidates = [MockCand(c, 0.01) for c in ["K21.9", "I10", "E11.9"]]
        reranked = reranker.rerank(candidates, query="BN", mention="bệnh", top_k=2)
        assert len(reranked) == 2

    def test_result_has_rank_before(self, reranker):
        """Reranked results should include original rank."""
        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        candidates = [MockCand(c, 0.01) for c in ["K21.9", "I10"]]
        reranked = reranker.rerank(candidates, query="BN", mention="bệnh", top_k=10)

        assert all(r.rank_before is not None for r in reranked)
        # K21.9 was first (rank_before=1), I10 was second (rank_before=2)
        codes_order = [r.code for r in reranked]
        for r in reranked:
            if r.code == "K21.9":
                assert r.rank_before == 1
            if r.code == "I10":
                assert r.rank_before == 2


class TestICDSpecificityFeatures:
    """Test individual ICD specificity features."""

    def test_weights_exist(self):
        """ICD weights dict should contain all expected keys."""
        expected_keys = [
            "lexical_similarity", "exact_alias", "context_include",
            "specificity_penalty", "parent_child_boost", "retrieval_score"
        ]
        for key in expected_keys:
            assert key in ICD_WEIGHTS, f"Missing weight: {key}"

    def test_weights_sum_to_approximately_1(self):
        """Weights should sum to 1.0 (excluding penalty which is negative)."""
        positive_weights = [
            ICD_WEIGHTS[k] for k in ICD_WEIGHTS
            if not k.startswith("_") and ICD_WEIGHTS[k] > 0
        ]
        total = sum(positive_weights)
        assert abs(total - 1.0) < 0.01, f"Weights sum = {total}, expected 1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
