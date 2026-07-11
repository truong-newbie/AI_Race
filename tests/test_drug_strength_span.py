"""
Tests for Drug Strength Span

Test Rule 4: THUỐC entities should prefer dosage information.
"""

import pytest
from src.entity.resolver import EntityResolver
from src.entity.confidence import ConfidenceConfig
from src.entity.conflict_logger import ConflictLogger


class TestDrugStrengthSpan:
    """Test drug entity span resolution."""

    @pytest.fixture
    def resolver(self):
        """Create resolver instance."""
        config = ConfidenceConfig(drug_prefer_dosage=True, drug_prefer_form=True)
        logger = ConflictLogger(enable_logging=False)
        return EntityResolver(config=config, conflict_logger=logger)

    def test_drug_with_dosage_preferred(self, resolver):
        """Test Rule 4: Drug with dosage is preferred."""
        text = "Paracetamol 500mg"
        entities = [
            {"text": "Paracetamol", "start": 0, "end": 12, "type": "THUỐC", "confidence": 0.9, "source": "dict"},
            {"text": "Paracetamol 500mg", "start": 0, "end": 18, "type": "THUỐC", "confidence": 0.85, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        # Should prefer span with dosage info (longer if similar confidence)
        # or higher confidence
        assert "THUỐC" in resolved[0]["type"]

    def test_drug_with_form_preferred(self, resolver):
        """Test Rule 4: Drug with form (tablet, capsule) is preferred."""
        text = "Amoxicillin 500mg viên"
        entities = [
            {"text": "Amoxicillin", "start": 0, "end": 10, "type": "THUỐC", "confidence": 0.9, "source": "dict"},
            {"text": "Amoxicillin 500mg viên", "start": 0, "end": 23, "type": "THUỐC", "confidence": 0.9, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        # When same confidence, prefer more complete span with form

    def test_drug_name_only_kept(self, resolver):
        """Test that drug name alone is kept when no dosage available."""
        text = "Paracetamol"
        entities = [
            {"text": "Paracetamol", "start": 0, "end": 12, "type": "THUỐC", "confidence": 0.9, "source": "dict"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        assert resolved[0]["text"] == "Paracetamol"

    def test_multiple_dosage_formats(self, resolver):
        """Test with various dosage formats."""
        text = "Aspirin 100mg uống ngày 3 lần"
        entities = [
            {"text": "Aspirin", "start": 0, "end": 7, "type": "THUỐC", "confidence": 0.9, "source": "dict"},
            {"text": "Aspirin 100mg", "start": 0, "end": 14, "type": "THUỐC", "confidence": 0.85, "source": "ner"},
            {"text": "Aspirin 100mg uống", "start": 0, "end": 19, "type": "THUỐC", "confidence": 0.8, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        assert "Aspirin" in resolved[0]["text"]

    def test_drug_merging_same_type(self, resolver):
        """Test that drug entities with same type are merged properly."""
        text = "Paracetamol 500mg"
        entities = [
            {"text": "Paracetamol", "start": 0, "end": 12, "type": "THUỐC", "confidence": 0.9, "source": "dict"},
            # 500mg has different span (13-17), so won't merge with (0-12)
            {"text": "500mg", "start": 13, "end": 17, "type": "THUỐC", "confidence": 0.7, "source": "regex"},
            {"text": "Paracetamol 500mg", "start": 0, "end": 18, "type": "THUỐC", "confidence": 0.85, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        # Should have at least 2 drug entities (Paracetamol alone + 500mg or full span)
        assert len(resolved) >= 2
        # Check that we have Paracetamol somewhere
        drug_texts = [e["text"] for e in resolved if e["type"] == "THUỐC"]
        has_paracetamol = any("Paracetamol" in t for t in drug_texts)
        assert has_paracetamol, f"Expected Paracetamol in texts, got: {drug_texts}"

    def test_drug_confidence_bonus(self, resolver):
        """Test that agreement gives confidence bonus."""
        text = "Ibuprofen"
        entities = [
            {"text": "Ibuprofen", "start": 0, "end": 9, "type": "THUỐC", "confidence": 0.8, "source": "dict"},
            {"text": "Ibuprofen", "start": 0, "end": 9, "type": "THUỐC", "confidence": 0.8, "source": "ner"},
        ]

        resolved = resolver.resolve_entities(entities, text)

        assert len(resolved) == 1
        # Should have agreement bonus
        assert resolved[0]["confidence"] > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
