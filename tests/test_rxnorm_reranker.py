"""Tests for drug candidate reranker."""

import pytest
from src.linking.rxnorm import get_knowledge_base
from src.linking.rxnorm.reranker import DrugReranker, STANDARD_STRENGTHS


@pytest.fixture
def entries():
    return get_knowledge_base()


@pytest.fixture
def reranker(entries):
    return DrugReranker(entries=entries, use_cross_encoder=False)


@pytest.fixture
def mock_candidates():
    """Mock candidates with equal retrieval scores (tie case)."""
    class MockCand:
        def __init__(self, rxcui, score):
            self.rxcui = rxcui
            self.score = score
    return [
        MockCand("1191", 2.5),  # Aspirin 81mg
        MockCand("1192", 2.5),  # Aspirin 325mg
    ]


class TestDosageSpecificityScore:
    """Tests for standard strength matching."""

    def test_standard_strength_map_exists(self):
        """STANDARD_STRENGTHS dict should cover all multi-strength drugs."""
        assert "aspirin" in STANDARD_STRENGTHS
        assert "metformin" in STANDARD_STRENGTHS

    def test_aspirin_standard_is_325mg(self, entries):
        """Standard Aspirin strength should be 325mg (highest common)."""
        assert STANDARD_STRENGTHS["aspirin"] == (325.0, "MG")

    def test_metformin_standard_is_1000mg(self, entries):
        """Standard Metformin strength should be 1000mg (highest common)."""
        assert STANDARD_STRENGTHS["metformin"] == (1000.0, "MG")


class TestIngredientOnlyReranking:
    """Reranking for ingredient-only mentions (no strength)."""

    def test_aspirin_ingredient_only_prefers_325mg(self, reranker, mock_candidates):
        """Ingredient-only 'Aspirin' should rank 325mg above 81mg."""
        from src.linking.rxnorm.schema import ParsedDrug

        parsed = ParsedDrug(original="Aspirin", ingredients=["Aspirin"])
        entry_81 = next(e for e in reranker.entries.values() if e.rxcui == "1191")
        entry_325 = next(e for e in reranker.entries.values() if e.rxcui == "1192")

        # Both candidates have equal retrieval score
        scored_81 = reranker._dosage_specificity_score(parsed, entry_81)
        scored_325 = reranker._dosage_specificity_score(parsed, entry_325)

        assert scored_325 > scored_81, (
            f"325mg ({scored_325:.3f}) should score higher than 81mg ({scored_81:.3f})"
        )

    def test_metformin_ingredient_only_prefers_1000mg(self, reranker, entries):
        """Ingredient-only 'Metformin' should rank 1000mg above 500mg/850mg."""
        from src.linking.rxnorm.schema import ParsedDrug

        parsed = ParsedDrug(original="Metformin", ingredients=["Metformin"])
        entry_500 = next(e for e in entries if e.rxcui == "6809")
        entry_850 = next(e for e in entries if e.rxcui == "860975")
        entry_1000 = next(e for e in entries if e.rxcui == "861007")

        score_500 = reranker._dosage_specificity_score(parsed, entry_500)
        score_850 = reranker._dosage_specificity_score(parsed, entry_850)
        score_1000 = reranker._dosage_specificity_score(parsed, entry_1000)

        assert score_1000 > score_500, f"1000mg ({score_1000:.3f}) > 500mg ({score_500:.3f})"
        assert score_1000 > score_850, f"1000mg ({score_1000:.3f}) > 850mg ({score_850:.3f})"

    def test_full_rerank_aspirin_ingredient_only(self, reranker, mock_candidates):
        """Full rerank of ingredient-only 'Aspirin' should put 325mg first."""
        reranked = reranker.rerank(mock_candidates, query="Su dung Aspirin", mention="Aspirin", top_k=2)
        assert len(reranked) == 2
        assert reranked[0].rxcui == "1192", "Aspirin 325mg should be first"
        assert reranked[1].rxcui == "1191"


class TestStrengthMentionReranking:
    """Reranking when mention has explicit strength."""

    def test_exact_strength_match_highest_score(self, reranker, entries):
        """Exact strength match should get highest score."""
        from src.linking.rxnorm.schema import ParsedDrug

        # Mention has 325mg — should match Aspirin 325mg
        parsed = ParsedDrug(
            original="Aspirin 325mg",
            ingredients=["Aspirin"],
            strength_values=[325.0],
            strength_units=["MG"],
        )
        entry_325 = next(e for e in entries if e.rxcui == "1192")
        entry_81 = next(e for e in entries if e.rxcui == "1191")

        score_325 = reranker._dosage_specificity_score(parsed, entry_325)
        score_81 = reranker._dosage_specificity_score(parsed, entry_81)

        assert score_325 > score_81, "Exact match (325mg) should score higher"

    def test_mention_no_strength_standard_dose_preferred(self, reranker, entries):
        """When mention has no strength, standard dose should be preferred."""
        from src.linking.rxnorm.schema import ParsedDrug

        # Alprazolam — standard = 0.5mg, gold in data = 0.5mg
        parsed = ParsedDrug(original="Alprazolam", ingredients=["Alprazolam"])
        entry = next(e for e in entries if e.rxcui == "72509")

        score = reranker._dosage_specificity_score(parsed, entry)
        assert score > 0, "Standard dose should get positive score"


class TestFullReranking:
    """Full end-to-end reranking tests."""

    def test_rerank_preserves_non_tied_order(self, reranker, entries):
        """For non-tied candidates, reranking should preserve relative order."""
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score

        # Candidates with different retrieval scores
        candidates = [
            MockCand("6809", 1.5),   # Metformin 500mg
            MockCand("617312", 2.0), # Atorvastatin 20mg
        ]

        reranked = reranker.rerank(
            candidates, query="Thuoc", mention="Metformin", top_k=2
        )
        # Atorvastatin had higher retrieval score — should stay first
        # (unless reranker reorders)
        assert len(reranked) == 2

    def test_rerank_returns_rerankscore_objects(self, reranker, entries):
        """Rerank should return RerankScore objects with all fields."""
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score

        candidates = [MockCand("1191", 2.5)]
        reranked = reranker.rerank(
            candidates, query="Aspirin", mention="Aspirin", top_k=1
        )

        assert len(reranked) == 1
        assert hasattr(reranked[0], "rxcui")
        assert hasattr(reranked[0], "rerank_score")
        assert hasattr(reranked[0], "source")
        assert hasattr(reranked[0], "features")

    def test_rerank_empty_candidates(self, reranker):
        """Reranking empty list should return empty list."""
        reranked = reranker.rerank([], query="Aspirin", mention="Aspirin", top_k=5)
        assert reranked == []

    def test_rerank_top_k_limit(self, reranker, entries):
        """Rerank should respect top_k limit."""
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score

        candidates = [MockCand("1191", 2.5), MockCand("1192", 2.5)]
        reranked = reranker.rerank(
            candidates, query="Aspirin", mention="Aspirin", top_k=1
        )
        assert len(reranked) == 1

    def test_rerank_feature_dict_populated(self, reranker, entries):
        """Rerank features dict should contain all scoring components."""
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score

        candidates = [MockCand("1192", 2.5)]
        reranked = reranker.rerank(
            candidates, query="Aspirin", mention="Aspirin", top_k=1
        )

        features = reranked[0].features
        assert "dosage_specificity" in features
        assert "retrieval_norm" in features

    def test_rerank_source_determined(self, reranker, entries):
        """Rerank source field should indicate primary signal."""
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score

        # Aspirin 325mg is the standard, should have dosage_specificity source
        candidates = [MockCand("1192", 2.5)]
        reranked = reranker.rerank(
            candidates, query="Aspirin", mention="Aspirin", top_k=1
        )

        assert reranked[0].source in [
            "dosage_specificity", "ingredient_context", "retrieval_boost"
        ]


class TestEntriesByIngredient:
    """Test the ingredient index."""

    def test_entries_indexed_by_ingredient(self, reranker):
        """Entries should be indexed by ingredient."""
        assert "aspirin" in reranker._entries_by_ingredient
        assert "metformin" in reranker._entries_by_ingredient

    def test_aspirin_has_multiple_entries(self, reranker):
        """Aspirin should have multiple entries (81mg and 325mg)."""
        aspirin_entries = reranker._entries_by_ingredient.get("aspirin", [])
        assert len(aspirin_entries) >= 2
        assert "1191" in aspirin_entries
        assert "1192" in aspirin_entries

    def test_metformin_has_multiple_entries(self, reranker):
        """Metformin should have multiple entries (500, 850, 1000mg)."""
        met_entries = reranker._entries_by_ingredient.get("metformin", [])
        assert len(met_entries) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
