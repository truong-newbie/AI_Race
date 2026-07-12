"""Tests for drug text normalizer."""

import pytest
from src.linking.rxnorm.normalizer import (
    DrugTextNormalizer,
    normalize_whitespace,
    normalize_dashes,
    expand_brand_aliases,
    expand_ingredient_synonyms,
)


class TestDrugTextNormalizer:
    """Test DrugTextNormalizer class."""

    @pytest.fixture
    def norm(self):
        return DrugTextNormalizer()

    def test_lowercase(self, norm):
        result = norm.normalize("ASPIRIN 81MG")
        assert result == "aspirin 81mg"

    def test_whitespace(self, norm):
        result = norm.normalize("Aspirin   81mg")
        assert "  " not in result

    def test_dash_normalization(self, norm):
        result = norm.normalize("viêm-phổi")
        assert "-" not in result

    def test_unit_decimal_comma(self, norm):
        result = norm._normalize_units("Aspirin 0,5 MG")
        assert "0.5" in result

    def test_brand_expansion(self, norm):
        result = norm.normalize("uống Xanax 0.5mg")
        assert "alprazolam" in result

    def test_synonym_expansion(self, norm):
        result = norm.normalize("Tylenol 500mg")
        assert "paracetamol" in result

    def test_empty(self, norm):
        result = norm.normalize("")
        assert result == ""

    def test_normalize_for_matching(self, norm):
        result = norm.normalize_for_matching("ASPIRIN 81MG")
        assert result == "aspirin 81mg"

    def test_ingredient_brand_synonyms(self, norm):
        """Brand names should expand to ingredient."""
        for brand, ingredient in [("lipitor", "atorvastatin"), ("plavix", "clopidogrel"),
                                   ("zoloft", "sertraline"), ("januvia", "sitagliptin")]:
            result = norm.normalize(f"{brand} 20mg")
            assert ingredient in result, f"{brand} should expand to {ingredient}"


class TestNormalizeWhitespace:
    def test_single_space(self):
        assert normalize_whitespace("Aspirin 81mg") == "Aspirin 81mg"

    def test_multiple_spaces(self):
        assert normalize_whitespace("Aspirin   81mg") == "Aspirin 81mg"

    def test_tabs_newlines(self):
        assert normalize_whitespace("Aspirin\t\n 81mg") == "Aspirin 81mg"


class TestNormalizeDashes:
    def test_hyphen(self):
        assert normalize_dashes("viêm-phổi") == "viêm phổi"

    def test_en_dash(self):
        assert normalize_dashes("viêm–phổi") == "viêm phổi"

    def test_em_dash(self):
        assert normalize_dashes("viêm—phổi") == "viêm phổi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
