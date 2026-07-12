"""Tests for drug mention parser."""

import pytest
from src.linking.rxnorm.parser import DrugMentionParser
from src.linking.rxnorm.schema import ParsedDrug


class TestParseStrength:
    """Test strength value and unit parsing."""

    @pytest.fixture
    def parser(self):
        return DrugMentionParser()

    def test_parse_mg(self, parser):
        """Parse MG strength."""
        result = parser.parse("Aspirin 81mg")
        assert result is not None
        assert result.strength_values == [81.0]
        assert result.strength_units == ["MG"]

    def test_parse_decimal_mg(self, parser):
        """Parse decimal MG strength (0.5mg)."""
        result = parser.parse("Alprazolam 0.5 MG")
        assert result is not None
        assert result.strength_values == [0.5]
        assert result.strength_units == ["MG"]

    def test_parse_gram(self, parser):
        """Parse gram strength."""
        result = parser.parse("Ceftriaxone 1g")
        assert result is not None
        assert result.strength_values == [1.0]
        assert result.strength_units == ["G"]

    def test_parse_gram_uppercase(self, parser):
        """Parse uppercase GM."""
        result = parser.parse("Ceftriaxone 1 GM")
        assert result is not None
        assert result.strength_values == [1.0]
        assert result.strength_units == ["G"]

    def test_parse_comma_decimal(self, parser):
        """Parse comma decimal (European format)."""
        result = parser.parse("Aspirin 0,5 MG")
        assert result is not None
        assert result.strength_values == [0.5]
        assert result.strength_units == ["MG"]

    def test_parse_no_strength(self, parser):
        """Drug mention without strength."""
        result = parser.parse("Metformin")
        assert result is not None
        assert result.strength_values == []
        assert result.strength_units == []

    def test_parse_multiple_doses(self, parser):
        """Multiple strength mentions in one text."""
        result = parser.parse("Aspirin 81mg + Ibuprofen 400mg")
        assert result is not None
        # Should parse at least one
        assert len(result.strength_values) >= 1


class TestParseIngredient:
    """Test ingredient extraction."""

    @pytest.fixture
    def parser(self):
        return DrugMentionParser()

    def test_simple_ingredient(self, parser):
        """Simple drug name."""
        result = parser.parse("Metformin 500mg")
        assert result is not None
        assert "metformin" in result.ingredients[0].lower()

    def test_ingredient_with_form(self, parser):
        """Ingredient with dose form."""
        result = parser.parse("Metformin 500mg tablet")
        assert result is not None
        assert result.main_ingredient() is not None
        assert result.dose_form == "tablet"

    def test_case_insensitive(self, parser):
        """Case insensitive parsing."""
        result1 = parser.parse("METFORMIN 500MG")
        result2 = parser.parse("metformin 500mg")
        assert result1 is not None
        assert result2 is not None
        assert result1.main_ingredient() is not None
        assert result1.main_ingredient().lower() == result2.main_ingredient().lower()

    def test_empty_input(self, parser):
        """Empty input returns None."""
        result = parser.parse("")
        assert result is None

    def test_ingredient_only(self, parser):
        """Ingredient without strength."""
        result = parser.parse("Aspirin")
        assert result is not None
        assert result.main_ingredient() is not None


class TestParseDoseForm:
    """Test dose form extraction."""

    @pytest.fixture
    def parser(self):
        return DrugMentionParser()

    def test_tablet(self, parser):
        result = parser.parse("Metformin 500mg tablet")
        assert result.dose_form == "tablet"

    def test_injection(self, parser):
        result = parser.parse("Ceftriaxone 1g injection")
        assert result.dose_form == "injection"

    def test_capsule(self, parser):
        result = parser.parse("Amoxicillin 500mg capsule")
        assert result.dose_form == "capsule"

    def test_no_form(self, parser):
        result = parser.parse("Aspirin 81mg")
        assert result.dose_form is None


class TestParsedDrugProperties:
    """Test ParsedDrug dataclass properties."""

    @pytest.fixture
    def parser(self):
        return DrugMentionParser()

    def test_has_strength(self, parser):
        result = parser.parse("Aspirin 81mg")
        assert result.has_strength() is True

    def test_no_strength(self, parser):
        result = parser.parse("Aspirin")
        assert result.has_strength() is False

    def test_is_combination(self, parser):
        result = parser.parse("Aspirin 81mg")
        assert result.is_combination() is False

    def test_main_ingredient(self, parser):
        result = parser.parse("Metformin 500mg")
        assert result.main_ingredient() is not None
        assert len(result.main_ingredient()) > 0

    def test_main_strength(self, parser):
        result = parser.parse("Aspirin 81mg")
        val, unit = result.main_strength()
        assert val == 81.0
        assert unit == "MG"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
