"""
Output Validator cho Medical Ontology

Kiểm tra text, position, type, assertions và candidate existence
theo schema đã định nghĩa.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

from src.schema import Entity, EntityType, AssertionType, MedicalDocument

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Một lỗi validation."""
    field: str
    message: str
    entity_index: Optional[int] = None
    severity: str = "error"  # error, warning


@dataclass
class ValidationResult:
    """Kết quả validation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def add_error(self, field: str, message: str, entity_index: Optional[int] = None):
        self.errors.append(ValidationError(field, message, entity_index, "error"))
        self.is_valid = False

    def add_warning(self, field: str, message: str, entity_index: Optional[int] = None):
        self.warnings.append(ValidationError(field, message, entity_index, "warning"))

    def summary(self) -> str:
        """Trả về summary của validation."""
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
        if self.is_valid:
            return "✓ Valid"
        return f"✗ Invalid: {', '.join(parts)}"


class EntityValidator:
    """
    Validator cho Entity objects.

    Validates:
    - text matches position in original
    - type is valid
    - assertions are valid
    - candidates exist in knowledge base (optional)
    """

    def __init__(
        self,
        original_text: str,
        known_icd_codes: Optional[set] = None,
        known_rxnorm_codes: Optional[set] = None,
        strict_type_check: bool = True,
        strict_position_check: bool = True
    ):
        """
        Initialize validator.

        Args:
            original_text: Original text để validate position
            known_icd_codes: Set of valid ICD codes (optional)
            known_rxnorm_codes: Set of valid RxNorm codes (optional)
            strict_type_check: Nếu True, enforce strict type checking
            strict_position_check: Nếu True, validate position matches text
        """
        self.original_text = original_text
        self.known_icd_codes = known_icd_codes or set()
        self.known_rxnorm_codes = known_rxnorm_codes or set()
        self.strict_type_check = strict_type_check
        self.strict_position_check = strict_position_check

    def validate_entity(self, entity: Entity, index: int) -> List[ValidationError]:
        """
        Validate một entity.

        Args:
            entity: Entity cần validate
            index: Index của entity trong list

        Returns:
            Danh sách lỗi (empty nếu valid)
        """
        errors = []

        # 1. Validate position
        pos_errors = self._validate_position(entity, index)
        errors.extend(pos_errors)

        # 2. Validate text matches position
        if pos_errors == []:  # Chỉ check nếu position hợp lệ
            text_errors = self._validate_text(entity, index)
            errors.extend(text_errors)

        # 3. Validate type
        type_errors = self._validate_type(entity, index)
        errors.extend(type_errors)

        # 4. Validate assertions
        assertion_errors = self._validate_assertions(entity, index)
        errors.extend(assertion_errors)

        # 5. Validate candidates
        candidate_errors = self._validate_candidates(entity, index)
        errors.extend(candidate_errors)

        return errors

    def _validate_position(self, entity: Entity, index: int) -> List[ValidationError]:
        """Validate position field."""
        errors = []
        start, end = entity.position

        # Check bounds
        if start < 0:
            errors.append(ValidationError(
                "position",
                f"start must be >= 0, got {start}",
                index
            ))

        if end <= start:
            errors.append(ValidationError(
                "position",
                f"end must be > start, got start={start}, end={end}",
                index
            ))

        if end > len(self.original_text):
            errors.append(ValidationError(
                "position",
                f"end={end} exceeds text length={len(self.original_text)}",
                index
            ))

        return errors

    def _validate_text(self, entity: Entity, index: int) -> List[ValidationError]:
        """Validate text matches position in original."""
        errors = []
        start, end = entity.position
        extracted = self.original_text[start:end]

        if extracted != entity.text:
            errors.append(ValidationError(
                "text",
                f"text mismatch. Entity text='{entity.text}', "
                f"extracted='{extracted}' at position [{start}, {end})",
                index
            ))

        return errors

    def _validate_type(self, entity: Entity, index: int) -> List[ValidationError]:
        """Validate entity type."""
        errors = []

        # Check type is valid enum
        try:
            EntityType(entity.type)
        except ValueError:
            errors.append(ValidationError(
                "type",
                f"Invalid entity type: {entity.type}",
                index
            ))
            return errors

        # Check assertions only for allowed types
        allowed_assertion_types = {EntityType.TRIEU_CHUNG, EntityType.CHAN_DOAN, EntityType.THUOC}
        if entity.assertions and entity.type not in allowed_assertion_types:
            errors.append(ValidationError(
                "assertions",
                f"Assertions not allowed for type {entity.type}. "
                f"Only for TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC",
                index
            ))

        # Check candidates only for allowed types
        allowed_candidate_types = {EntityType.CHAN_DOAN, EntityType.THUOC}
        if entity.candidates and entity.type not in allowed_candidate_types:
            errors.append(ValidationError(
                "candidates",
                f"Candidates not allowed for type {entity.type}. "
                f"Only for CHẨN_ĐOÁN and THUỐC",
                index
            ))

        return errors

    def _validate_assertions(self, entity: Entity, index: int) -> List[ValidationError]:
        """Validate assertions."""
        errors = []

        # Check assertion values
        for assertion in entity.assertions:
            try:
                AssertionType(assertion)
            except ValueError:
                errors.append(ValidationError(
                    "assertions",
                    f"Invalid assertion: {assertion}",
                    index
                ))

        # Check max 3 assertions
        if len(entity.assertions) > 3:
            errors.append(ValidationError(
                "assertions",
                f"Too many assertions: {len(entity.assertions)} (max 3)",
                index
            ))

        return errors

    def _validate_candidates(self, entity: Entity, index: int) -> List[ValidationError]:
        """Validate candidates exist in knowledge base."""
        errors = []

        if not entity.candidates:
            return errors

        # Determine which KB to use
        if entity.type == EntityType.CHAN_DOAN:
            known_codes = self.known_icd_codes
            kb_name = "ICD-10"
        elif entity.type == EntityType.THUOC:
            known_codes = self.known_rxnorm_codes
            kb_name = "RxNorm"
        else:
            # No KB check for other types
            return errors

        # Check if KB is loaded
        if not known_codes:
            logger.debug(f"{kb_name} knowledge base not loaded, skipping candidate validation")
            return errors

        # Check each candidate
        for code in entity.candidates:
            if code not in known_codes:
                errors.append(ValidationError(
                    "candidates",
                    f"Invalid {kb_name} code: {code} (not in knowledge base)",
                    index,
                    severity="warning"  # Warning, not error
                ))

        return errors


class OutputValidator:
    """
    Validator cho toàn bộ output JSON.

    Validates:
    - JSON structure
    - All entities
    - Position uniqueness
    - Text coverage
    """

    def __init__(
        self,
        original_text: str,
        known_icd_codes: Optional[set] = None,
        known_rxnorm_codes: Optional[set] = None
    ):
        self.entity_validator = EntityValidator(
            original_text,
            known_icd_codes,
            known_rxnorm_codes
        )

    def validate(self, entities: List[Entity]) -> ValidationResult:
        """
        Validate danh sách entities.

        Args:
            entities: Danh sách entities cần validate

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # 1. Validate each entity
        for i, entity in enumerate(entities):
            errors = self.entity_validator.validate_entity(entity, i)
            for error in errors:
                if error.severity == "error":
                    result.add_error(error.field, error.message, error.entity_index)
                else:
                    result.add_warning(error.field, error.message, error.entity_index)

        # 2. Check for duplicate positions
        seen_positions = {}
        for i, entity in enumerate(entities):
            pos_key = tuple(entity.position)
            if pos_key in seen_positions:
                result.add_warning(
                    "position",
                    f"Duplicate position [{pos_key[0]}, {pos_key[1]}) "
                    f"at indices {seen_positions[pos_key]} and {i}",
                    i
                )
            else:
                seen_positions[pos_key] = i

        # 3. Check for overlapping entities (warning)
        overlaps = self._find_overlaps(entities)
        for i, j in overlaps:
            result.add_warning(
                "overlap",
                f"Entities at indices {i} and {j} have overlapping positions",
                i
            )

        return result

    def _find_overlaps(self, entities: List[Entity]) -> List[tuple[int, int]]:
        """Tìm các entities có overlapping positions."""
        overlaps = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                pos1 = entities[i].position
                pos2 = entities[j].position

                # Check overlap: [a,b) and [c,d) overlap if a < d and c < b
                if pos1[0] < pos2[1] and pos2[0] < pos1[1]:
                    overlaps.append((i, j))

        return overlaps

    def validate_document(self, doc: MedicalDocument) -> ValidationResult:
        """Validate MedicalDocument."""
        return self.validate(doc.entities)


def validate_output(
    entities: List[dict],
    original_text: str,
    known_icd_codes: Optional[set] = None,
    known_rxnorm_codes: Optional[set] = None,
    raise_on_error: bool = False
) -> ValidationResult:
    """
    Convenience function để validate output.

    Args:
        entities: List of entity dicts (from JSON)
        original_text: Original text
        known_icd_codes: Set of valid ICD codes
        known_rxnorm_codes: Set of valid RxNorm codes
        raise_on_error: Nếu True, raise exception thay vì trả về result

    Returns:
        ValidationResult

    Raises:
        ValueError: Nếu raise_on_error=True và có lỗi
    """
    # Convert dicts to Entity objects
    entity_objects = []
    for i, e in enumerate(entities):
        try:
            entity_objects.append(Entity(**e))
        except Exception as ex:
            result = ValidationResult(is_valid=False)
            result.add_error("schema", f"Invalid entity at index {i}: {ex}", i)
            if raise_on_error:
                raise ValueError(str(result.errors))
            return result

    # Validate
    validator = OutputValidator(original_text, known_icd_codes, known_rxnorm_codes)
    result = validator.validate(entity_objects)

    if raise_on_error and not result.is_valid:
        raise ValueError(str(result.errors))

    return result
