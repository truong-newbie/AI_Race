"""
Tests cho validation module.
"""

import pytest
from src.schema import Entity, EntityType, AssertionType
from src.validation.validator import (
    EntityValidator,
    OutputValidator,
    ValidationResult,
    validate_output,
)


class TestEntityValidator:
    """Tests cho EntityValidator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.original_text = "BN ho đờm, tức ngực."
        self.validator = EntityValidator(self.original_text)

    def test_valid_entity(self):
        """Test validate entity hợp lệ."""
        entity = Entity(
            text="ho đờm",
            position=[3, 9],
            type=EntityType.TRIEU_CHUNG
        )
        errors = self.validator.validate_entity(entity, 0)
        assert errors == []

    def test_invalid_position_exceeds_length(self):
        """Test position vượt quá text length."""
        entity = Entity(
            text="test",
            position=[0, 100],
            type=EntityType.TRIEU_CHUNG
        )
        errors = self.validator.validate_entity(entity, 0)
        assert len(errors) >= 1
        assert any("exceeds" in e.message for e in errors)

    def test_text_mismatch(self):
        """Test text không match position."""
        entity = Entity(
            text="không khớp",
            position=[0, 5],
            type=EntityType.TRIEU_CHUNG
        )
        errors = self.validator.validate_entity(entity, 0)
        assert len(errors) >= 1
        assert any("mismatch" in e.message.lower() for e in errors)

    def test_invalid_entity_type(self):
        """Test invalid entity type."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            Entity(
                text="test",
                position=[0, 4],
                type="INVALID_TYPE"
            )

    def test_candidates_for_wrong_type(self):
        """Test candidates cho type không hỗ trợ."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            Entity(
                text="WBC",
                position=[0, 3],
                type=EntityType.TEN_XET_NGHIEM,
                candidates=["some_code"]
            )

    def test_assertions_for_wrong_type(self):
        """Test assertions cho type không hỗ trợ."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            Entity(
                text="WBC",
                position=[0, 3],
                type=EntityType.TEN_XET_NGHIEM,
                assertions=[AssertionType.HISTORICAL]
            )

    def test_candidates_with_known_codes(self):
        """Test candidates với known codes."""
        validator = EntityValidator(
            self.original_text,
            known_icd_codes={"K21.0", "K21.9"}
        )
        entity = Entity(
            text="trào",
            position=[0, 4],
            type=EntityType.CHAN_DOAN,
            candidates=["K21.0"]
        )
        errors = validator.validate_entity(entity, 0)
        # No error for valid code
        assert not any("Invalid" in e.message for e in errors)

    def test_candidates_with_unknown_codes(self):
        """Test candidates với unknown codes."""
        validator = EntityValidator(
            self.original_text,
            known_icd_codes={"K21.0"}
        )
        entity = Entity(
            text="trào",
            position=[0, 4],
            type=EntityType.CHAN_DOAN,
            candidates=["X99.9"]  # Not in known codes
        )
        errors = validator.validate_entity(entity, 0)
        # Warning, not error
        assert any("Invalid" in e.message and e.severity == "warning" for e in errors)


class TestOutputValidator:
    """Tests cho OutputValidator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.original_text = "BN ho đờm, tức ngực."
        self.validator = OutputValidator(self.original_text)

    def test_valid_output(self):
        """Test validate output hợp lệ."""
        entities = [
            Entity(
                text="ho đờm",
                position=[3, 9],
                type=EntityType.TRIEU_CHUNG
            ),
            Entity(
                text="tức ngực",
                position=[11, 19],
                type=EntityType.TRIEU_CHUNG
            )
        ]
        result = self.validator.validate(entities)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_duplicate_position_warning(self):
        """Test warning cho duplicate positions."""
        entities = [
            Entity(
                text="ho",
                position=[3, 5],
                type=EntityType.TRIEU_CHUNG
            ),
            Entity(
                text="ho",
                position=[3, 5],
                type=EntityType.TRIEU_CHUNG
            )
        ]
        result = self.validator.validate(entities)
        # Should have warnings but be valid
        assert len(result.warnings) >= 1
        assert any("Duplicate" in w.message for w in result.warnings)

    def test_overlapping_entities(self):
        """Test overlapping entities."""
        entities = [
            Entity(
                text="ho đ",
                position=[3, 7],
                type=EntityType.TRIEU_CHUNG
            ),
            Entity(
                text="đờm",
                position=[6, 9],
                type=EntityType.TRIEU_CHUNG
            )
        ]
        result = self.validator.validate(entities)
        # Should have overlap warnings
        assert len(result.warnings) >= 1
        assert any("overlap" in w.message.lower() for w in result.warnings)


class TestValidateOutput:
    """Tests cho convenience function."""

    def test_dict_to_entity_conversion(self):
        """Test convert dicts to Entity."""
        entities = [
            {
                "text": "ho đờm",
                "position": [3, 9],
                "type": "TRIỆU_CHỨNG",
                "assertions": [],
                "candidates": []
            }
        ]
        original_text = "BN ho đờm, tức ngực."
        result = validate_output(entities, original_text)
        assert result.is_valid is True

    def test_invalid_dict_schema(self):
        """Test invalid dict schema."""
        entities = [
            {
                "text": "test",
                # missing required fields
            }
        ]
        original_text = "test"
        result = validate_output(entities, original_text)
        assert result.is_valid is False
        assert len(result.errors) >= 1

    def test_raise_on_error(self):
        """Test raise_on_error parameter."""
        entities = [{"text": "test"}]
        with pytest.raises(ValueError):
            validate_output(entities, "test", raise_on_error=True)


class TestValidationResult:
    """Tests cho ValidationResult."""

    def test_add_error(self):
        """Test thêm error."""
        result = ValidationResult(is_valid=True)
        result.add_error("field", "message", 0)
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_add_warning(self):
        """Test thêm warning."""
        result = ValidationResult(is_valid=True)
        result.add_warning("field", "message", 0)
        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_summary_valid(self):
        """Test summary khi valid."""
        result = ValidationResult(is_valid=True)
        assert result.summary() == "✓ Valid"

    def test_summary_invalid(self):
        """Test summary khi invalid."""
        result = ValidationResult(is_valid=True)
        result.add_error("field", "message", 0)
        result.add_warning("field", "message", 0)
        summary = result.summary()
        assert "✗ Invalid" in summary
        assert "1 errors" in summary
        assert "1 warnings" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
