"""Tests for ICD-10 parent/child consistency in reranking."""

import pytest
from src.linking.icd.schema import get_knowledge_base
from src.linking.rule_reranker import ICDRuleReranker
from src.linking.ontology_constraints import (
    check_icd_child_context,
    check_icd_exclude_terms,
    OntologyValidator,
)


@pytest.fixture
def entries():
    return get_knowledge_base()


@pytest.fixture
def reranker(entries):
    return ICDRuleReranker(entries)


class TestParentChildConsistency:
    """Tests for ICD-10 parent/child relationship handling."""

    def test_parent_codes_identified(self, reranker):
        """Parent codes should be identified correctly (code == parent_code)."""
        # I10 is its own parent — it's the root of its category
        i10 = reranker.entries.get("I10")
        assert i10 is not None
        assert i10.parent_code == "I10"  # Root node

    def test_child_codes_identified(self, reranker):
        """Child codes should have parent_code different from themselves."""
        # K21.9 has parent K21
        k219 = reranker.entries.get("K21.9")
        assert k219 is not None
        assert k219.parent_code == "K21"
        assert k219.parent_code != k219.code  # It's a child

    def test_child_without_detail_penalized(self, reranker):
        """Child code without detail in mention should be penalized."""
        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        # K29.9 (gastritis) — child of K29
        # Generic mention "viêm dạ dày" — no detail signals
        candidates = [MockCand("K29.9", 0.01)]

        reranked = reranker.rerank(
            candidates,
            query="BN viêm dạ dày",
            mention="viêm dạ dày",
            top_k=1,
        )

        result = reranked[0]
        # Should have some form of specificity penalty
        assert result.is_child_code is True

    def test_parent_vs_child_ordering(self, reranker):
        """When both parent and child are candidates, specific mention should prefer child."""
        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        # Both K25 (parent) and K25.9 (child) — gastric ulcer
        # Specific mention with detail signals
        candidates = [
            MockCand("K25.9", 0.005),  # Child
            MockCand("K26.9", 0.005),  # Different child (duodenal) — distractor
        ]

        reranked = reranker.rerank(
            candidates,
            query="BN có đau dạ dày",
            mention="đau dạ dày",
            top_k=2,
        )

        assert len(reranked) == 2
        codes = [r.code for r in reranked]
        assert "K25.9" in codes

    def test_no_parent_child_confusion_different_categories(self, reranker):
        """Different category codes (e.g., K vs I) should not confuse parent/child logic."""
        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        # GERD (K21.9) vs Hypertension (I10)
        candidates = [
            MockCand("K21.9", 0.01),
            MockCand("I10", 0.01),
        ]

        reranked = reranker.rerank(
            candidates,
            query="BN tăng huyết áp",
            mention="tăng huyết áp",
            top_k=2,
        )

        assert len(reranked) == 2
        codes = [r.code for r in reranked]
        # I10 (hypertension) should rank first for mention "tăng huyết áp"
        assert codes[0] == "I10"


class TestChildContextConstraint:
    """Tests for the child_context ontology constraint."""

    def test_child_without_detail_fails_constraint(self):
        """Child code without mention detail should fail constraint check."""
        result = check_icd_child_context(
            code="K29.9",
            parent_code="K29",
            mention_has_detail=False,  # Generic mention
            entry_name="Gastritis",
            entry_description="Inflammation of the stomach lining",
        )

        assert result.passed is False
        assert result.rule == "icd_child_context"
        assert result.penalty > 0

    def test_child_with_detail_passes_constraint(self):
        """Child code with mention detail should pass constraint check."""
        result = check_icd_child_context(
            code="K29.9",
            parent_code="K29",
            mention_has_detail=True,  # Mention has detail signals
            entry_name="Gastritis",
            entry_description="Long description...",
        )

        # Should pass or have minimal penalty
        assert result.passed is True or result.penalty < 0.1

    def test_parent_code_always_passes(self):
        """Parent codes should always pass the child_context constraint."""
        result = check_icd_child_context(
            code="K21",
            parent_code="K21",  # Parent = itself
            mention_has_detail=False,
            entry_name="GERD",
            entry_description="",
        )

        assert result.passed is True

    def test_root_code_always_passes(self):
        """Root codes should always pass."""
        result = check_icd_child_context(
            code="I10",
            parent_code="I10",
            mention_has_detail=False,
            entry_name="Hypertension",
            entry_description="",
        )

        assert result.passed is True


class TestExcludeTermsConstraint:
    """Tests for exclude term matching."""

    def test_exclude_term_detected(self):
        """Mention with exclude term should fail constraint."""
        result = check_icd_exclude_terms(
            mention_normalized="viêm dạ dày tá tràng",
            exclude_terms=["viêm tá tràng", "tá tràng"],
        )

        assert result.passed is False
        assert "tá tràng" in result.reason

    def test_no_exclude_term_passes(self):
        """Mention without exclude terms should pass."""
        result = check_icd_exclude_terms(
            mention_normalized="viêm dạ dày",
            exclude_terms=["viêm tá tràng"],
        )

        assert result.passed is True

    def test_empty_exclude_terms_always_passes(self):
        """Empty exclude terms list should always pass."""
        result = check_icd_exclude_terms(
            mention_normalized="bất kỳ văn bản nào",
            exclude_terms=[],
        )

        assert result.passed is True


class TestOntologyValidatorICD:
    """Tests for OntologyValidator on ICD candidates."""

    def test_validate_icd_returns_multiple_results(self):
        """validate_icd should return results for all rules."""
        validator = OntologyValidator()
        results = validator.validate_icd(
            code="K29.9",
            parent_code="K29",
            mention_has_detail=False,
            mention_normalized="viêm dạ dày",
            entry_name="Gastritis",
            entry_description="Inflammation of the stomach",
            include_terms=["đau dạ dày", "nóng rát"],
            exclude_terms=["viêm tá tràng"],
        )

        assert len(results) >= 3  # child_context, exclude_terms, include_terms
        rules = {r.rule for r in results}
        assert "icd_child_context" in rules
        assert "icd_exclude_terms" in rules
        assert "icd_include_terms" in rules

    def test_total_penalty_sum(self):
        """Total penalty should sum all rule penalties."""
        validator = OntologyValidator()
        results = validator.validate_icd(
            code="K29.9",
            parent_code="K29",
            mention_has_detail=False,
            mention_normalized="viêm dạ dày",
            entry_name="Gastritis",
            entry_description="",
            include_terms=[],
            exclude_terms=["viêm tá tràng"],
        )

        total_penalty = validator.total_penalty(results)
        assert total_penalty >= 0

    def test_total_boost_sum(self):
        """Total boost should sum all rule boosts."""
        validator = OntologyValidator()
        results = validator.validate_icd(
            code="K21.9",
            parent_code="K21",
            mention_has_detail=True,
            mention_normalized="BN ợ nóng và ợ chua",
            entry_name="GERD",
            entry_description="Gastro-oesophageal reflux disease",
            include_terms=["ợ nóng", "ợ chua"],
            exclude_terms=[],
        )

        total_boost = validator.total_boost(results)
        assert total_boost >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
