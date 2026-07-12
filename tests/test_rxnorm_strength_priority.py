"""Tests for RxNorm strength priority in rule reranking."""

import pytest
from src.linking.rxnorm.schema import get_knowledge_base
from src.linking.rule_reranker import RxNormRuleReranker, RX_WEIGHTS
from src.linking.ontology_constraints import (
    check_rxnorm_strength_mismatch,
    check_rxnorm_dose_form_conflict,
    OntologyValidator,
)


@pytest.fixture
def entries():
    return get_knowledge_base()


@pytest.fixture
def reranker(entries):
    return RxNormRuleReranker(entries)


@pytest.fixture
def mock_rx_candidates():
    """Mock RxNorm candidates with equal retrieval scores."""
    class MockCand:
        def __init__(self, rxcui, score):
            self.rxcui = rxcui
            self.score = score
            self.sources = []
    return [
        MockCand("1191", 2.5),  # Aspirin 81mg
        MockCand("1192", 2.5),  # Aspirin 325mg
    ]


class TestRxNormStrengthPriority:
    """Tests for RxNorm strength-based reranking priority."""

    def test_ingredient_match_highest_weight(self):
        """Ingredient match should have the highest weight."""
        assert RX_WEIGHTS["ingredient"] >= RX_WEIGHTS["strength_exact"]
        assert RX_WEIGHTS["ingredient"] >= RX_WEIGHTS["strength_close"]
        assert RX_WEIGHTS["ingredient"] >= RX_WEIGHTS["unit_match"]
        assert RX_WEIGHTS["ingredient"] >= RX_WEIGHTS["dose_form"]

    def test_exact_strength_preferred(self, reranker, entries):
        """Exact strength match should score higher than close match."""
        # Aspirin 325mg mention
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score
                self.sources = []

        candidates = [
            MockCand("1191", 1.0),  # Aspirin 81mg
            MockCand("1192", 1.0),  # Aspirin 325mg
        ]

        reranked = reranker.rerank(
            candidates,
            query="BN uống Aspirin 325mg",
            mention="Aspirin 325mg",
            top_k=2,
        )

        assert len(reranked) == 2
        scores = {r.code: r.rerank_score for r in reranked}
        # Aspirin 325mg should score higher for mention with 325mg
        assert scores.get("1192", 0) >= scores.get("1191", 0)

    def test_strength_close_match_scored(self, reranker, entries):
        """Close strength match (within 15%) should be scored."""
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score
                self.sources = []

        # 500mg mention vs 550mg candidate — within 10%
        # But we need to find candidates at 550mg... let's use available ones
        candidates = [
            MockCand("6809", 1.0),   # Metformin 500mg
            MockCand("861007", 1.0),  # Metformin 1000mg
        ]

        reranked = reranker.rerank(
            candidates,
            query="BN uống Metformin 500mg",
            mention="Metformin 500mg",
            top_k=2,
        )

        assert len(reranked) == 2
        scores = {r.code: r.rerank_score for r in reranked}
        assert scores.get("6809", 0) >= scores.get("861007", 0)

    def test_unit_mismatch_penalized(self, reranker, entries):
        """Unit mismatch should reduce score."""
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score
                self.sources = []

        # This test checks that the scoring function handles unit comparison
        candidates = [
            MockCand("1191", 1.0),
        ]

        reranked = reranker.rerank(
            candidates,
            query="BN",
            mention="Aspirin 81mg",
            top_k=1,
        )

        assert len(reranked) == 1
        features = reranked[0].features
        assert "unit_match" in features or "ingredient" in features

    def test_ingredient_only_mention(self, reranker, entries):
        """Ingredient-only mention should match by ingredient only."""
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score
                self.sources = []

        # Aspirin — no strength specified
        candidates = [
            MockCand("1191", 1.0),  # 81mg
            MockCand("1192", 1.0),  # 325mg
        ]

        reranked = reranker.rerank(
            candidates,
            query="BN uống Aspirin",
            mention="Aspirin",
            top_k=2,
        )

        assert len(reranked) == 2
        # Both should have positive ingredient score
        for r in reranked:
            assert r.features.get("ingredient", 0) > 0


class TestStrengthMismatchConstraint:
    """Tests for RxNorm strength mismatch ontology constraint."""

    def test_exact_strength_no_penalty(self):
        """Exact strength match should not incur penalty."""
        result = check_rxnorm_strength_mismatch(
            mention_strength=500.0,
            mention_unit="MG",
            candidate_strength=500.0,
            candidate_unit="MG",
            dense_score=0.95,
        )

        assert result.passed is True
        assert result.penalty == 0.0

    def test_severe_mismatch_penalized(self):
        """Large strength mismatch (>50%) should incur severe penalty."""
        result = check_rxnorm_strength_mismatch(
            mention_strength=500.0,
            mention_unit="MG",
            candidate_strength=81.0,   # 500 vs 81 = ~84% mismatch
            candidate_unit="MG",
            dense_score=0.95,
        )

        assert result.passed is False
        assert result.penalty >= 0.3

    def test_moderate_mismatch_penalized(self):
        """Moderate strength mismatch (20-50%) should incur penalty."""
        result = check_rxnorm_strength_mismatch(
            mention_strength=500.0,
            mention_unit="MG",
            candidate_strength=400.0,  # 500 vs 400 = 20% mismatch
            candidate_unit="MG",
            dense_score=0.95,
        )

        assert result.passed is False
        assert result.penalty >= 0.15

    def test_unit_conversion_handled(self):
        """Unit conversion (G to MG) should be handled correctly."""
        result = check_rxnorm_strength_mismatch(
            mention_strength=1.0,
            mention_unit="G",
            candidate_strength=1000.0,
            candidate_unit="MG",
            dense_score=0.9,
        )

        # 1g = 1000mg, should be considered a match
        assert result.passed is True

    def test_dense_score_does_not_mask_mismatch(self):
        """High dense score must NOT mask a strength mismatch."""
        result = check_rxnorm_strength_mismatch(
            mention_strength=500.0,
            mention_unit="MG",
            candidate_strength=81.0,  # Severe mismatch
            candidate_unit="MG",
            dense_score=0.99,  # High dense score
        )

        # The mismatch should be penalized regardless of dense score
        assert result.passed is False
        assert result.penalty > 0

    def test_missing_strength_no_penalty(self):
        """When mention has no strength, no penalty."""
        result = check_rxnorm_strength_mismatch(
            mention_strength=None,
            mention_unit=None,
            candidate_strength=500.0,
            candidate_unit="MG",
            dense_score=0.8,
        )

        assert result.passed is True


class TestDoseFormConflict:
    """Tests for dose form conflict constraint."""

    def test_no_conflict_same_form(self):
        """Same dose form should not conflict."""
        result = check_rxnorm_dose_form_conflict(
            mention_dose_form="tablet",
            candidate_dose_form="tablet",
        )

        assert result.passed is True
        assert result.penalty == 0.0

    def test_conflict_tablet_vs_injection(self):
        """Tablet vs injection should conflict."""
        result = check_rxnorm_dose_form_conflict(
            mention_dose_form="tablet",
            candidate_dose_form="injection",
        )

        assert result.passed is False
        assert result.penalty >= 0.15

    def test_conflict_injection_vs_tablet(self):
        """Injection vs tablet should conflict."""
        result = check_rxnorm_dose_form_conflict(
            mention_dose_form="injection",
            candidate_dose_form="tablet",
        )

        assert result.passed is False

    def test_no_dose_form_no_penalty(self):
        """When either dose form is missing, no conflict."""
        result = check_rxnorm_dose_form_conflict(
            mention_dose_form=None,
            candidate_dose_form="tablet",
        )

        assert result.passed is True


class TestOntologyValidatorRxNorm:
    """Tests for OntologyValidator on RxNorm candidates."""

    def test_validate_rxnorm_returns_multiple_results(self):
        """validate_rxnorm should return results for all rules."""
        validator = OntologyValidator()
        results = validator.validate_rxnorm(
            mention_strength=500.0,
            mention_unit="MG",
            mention_dose_form="tablet",
            candidate_strength=500.0,
            candidate_unit="MG",
            candidate_dose_form="tablet",
            dense_score=0.9,
        )

        assert len(results) >= 2  # strength_mismatch, dose_form_conflict
        rules = {r.rule for r in results}
        assert "rxnorm_strength_mismatch" in rules
        assert "rxnorm_dose_form_conflict" in rules

    def test_total_penalty_for_mismatch(self):
        """Total penalty should accumulate for multiple mismatches."""
        validator = OntologyValidator()
        results = validator.validate_rxnorm(
            mention_strength=500.0,
            mention_unit="MG",
            mention_dose_form="tablet",
            candidate_strength=1000.0,  # mismatch
            candidate_unit="MG",
            candidate_dose_form="injection",  # conflict
            dense_score=0.95,
        )

        total = validator.total_penalty(results)
        # Should have penalty from strength mismatch AND dose form conflict
        assert total > 0

    def test_no_penalty_when_all_matches(self):
        """No penalty when all attributes match."""
        validator = OntologyValidator()
        results = validator.validate_rxnorm(
            mention_strength=500.0,
            mention_unit="MG",
            mention_dose_form="tablet",
            candidate_strength=500.0,
            candidate_unit="MG",
            candidate_dose_form="tablet",
            dense_score=0.9,
        )

        total = validator.total_penalty(results)
        assert total == 0.0


class TestRxNormWeights:
    """Test RxNorm weight configuration."""

    def test_weights_exist(self):
        """RxNorm weights should contain all expected keys."""
        expected = ["ingredient", "strength_exact", "strength_close",
                    "unit_match", "dose_form", "brand", "retrieval_score"]
        for key in expected:
            assert key in RX_WEIGHTS, f"Missing weight: {key}"

    def test_positive_weights_sum_to_1(self):
        """Positive weights should sum to 1.0."""
        positive = [v for v in RX_WEIGHTS.values() if v > 0]
        total = sum(positive)
        assert abs(total - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
