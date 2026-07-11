"""
Data Validators

Validators cho synthetic data:
- Schema validation
- Span validation
- Entity validation
- Duplicate detection
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .schema import Sample, Entity, EntityType, AssertionType, VALID_ENTITY_TYPES, VALID_ASSERTION_TYPES


# =============================================================================
# Validation Result
# =============================================================================

@dataclass
class ValidationError:
    """Một validation error."""
    sample_id: str
    field: str
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationResult:
    """Kết quả validation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def add_error(self, sample_id: str, field: str, message: str):
        self.errors.append(ValidationError(sample_id, field, message, "error"))
        self.is_valid = False

    def add_warning(self, sample_id: str, field: str, message: str):
        self.warnings.append(ValidationError(sample_id, field, message, "warning"))

    def summary(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [e.message for e in self.errors],
            "warnings": [w.message for w in self.warnings],
        }


# =============================================================================
# Individual Validators
# =============================================================================

class SchemaValidator:
    """Validate sample schema."""

    def validate(self, sample: Sample) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        # Check required fields
        if not sample.id:
            result.add_error(sample.id, "id", "Sample ID is required")

        if not sample.text:
            result.add_error(sample.id, "text", "Text is required")

        if not sample.entities:
            result.add_warning(sample.id, "entities", "Sample has no entities")

        return result


class SpanValidator:
    """Validate entity spans."""

    def validate(self, sample: Sample) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        if not sample.text:
            return result

        for i, entity in enumerate(sample.entities):
            # Check span bounds
            if entity.start < 0:
                result.add_error(
                    sample.id,
                    f"entities[{i}].start",
                    f"Invalid start position: {entity.start} < 0"
                )

            if entity.end > len(sample.text):
                result.add_error(
                    sample.id,
                    f"entities[{i}].end",
                    f"End position {entity.end} exceeds text length {len(sample.text)}"
                )

            if entity.start >= entity.end:
                result.add_error(
                    sample.id,
                    f"entities[{i}]",
                    f"Invalid span: start ({entity.start}) >= end ({entity.end})"
                )

            # Check span text matches
            if 0 <= entity.start < entity.end <= len(sample.text):
                extracted_text = sample.text[entity.start:entity.end]
                if extracted_text != entity.text:
                    result.add_error(
                        sample.id,
                        f"entities[{i}]",
                        f"Text mismatch: expected '{entity.text}', got '{extracted_text}'"
                    )

        return result


class EntityTypeValidator:
    """Validate entity types."""

    def validate(self, sample: Sample) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        for i, entity in enumerate(sample.entities):
            if entity.type not in VALID_ENTITY_TYPES:
                result.add_error(
                    sample.id,
                    f"entities[{i}].type",
                    f"Invalid entity type: '{entity.type}'. Valid types: {VALID_ENTITY_TYPES}"
                )

        return result


class AssertionValidator:
    """Validate assertions."""

    def validate(self, sample: Sample) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        for i, entity in enumerate(sample.entities):
            for assertion in entity.assertions:
                if assertion not in VALID_ASSERTION_TYPES:
                    result.add_error(
                        sample.id,
                        f"entities[{i}].assertions",
                        f"Invalid assertion: '{assertion}'. Valid types: {VALID_ASSERTION_TYPES}"
                    )

        return result


class PositionConventionValidator:
    """
    Validate position convention is [start, end) (end-exclusive).

    This is the standard convention used in the project.
    """

    def validate(self, sample: Sample) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        # Check no span touches or exceeds text boundary oddly
        for i, entity in enumerate(sample.entities):
            # Verify no entity spans are zero-width
            if entity.start == entity.end:
                result.add_error(
                    sample.id,
                    f"entities[{i}]",
                    "Zero-width entity span"
                )

        return result


class CandidateValidator:
    """Validate linking candidates."""

    def validate(self, sample: Sample) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        for i, entity in enumerate(sample.entities):
            # For CHẨN_ĐOÁN, candidates should be ICD-10 codes
            if entity.type == EntityType.CHAN_DOAN.value:
                for j, candidate in enumerate(entity.candidates):
                    # Basic ICD-10 format check
                    if not self._looks_like_icd10(candidate):
                        result.add_warning(
                            sample.id,
                            f"entities[{i}].candidates[{j}]",
                            f"Candidate '{candidate}' may not be a valid ICD-10 code"
                        )

            # For THUỐC, candidates should be RxCUI
            elif entity.type == EntityType.THUOC.value:
                for j, candidate in enumerate(entity.candidates):
                    if not self._looks_like_rxcui(candidate):
                        result.add_warning(
                            sample.id,
                            f"entities[{i}].candidates[{j}]",
                            f"Candidate '{candidate}' may not be a valid RxCUI"
                        )

        return result

    def _looks_like_icd10(self, code: str) -> bool:
        """Check if code looks like ICD-10 format."""
        # ICD-10 format: letter + 2 digits, optionally . + digits
        pattern = r'^[A-Z]\d{2}(\.\d+)?$'
        return bool(re.match(pattern, code))

    def _looks_like_rxcui(self, code: str) -> bool:
        """Check if code looks like RxCUI format."""
        # RxCUI format: numeric string
        return code.isdigit()


# =============================================================================
# Composite Validator
# =============================================================================

class DataValidator:
    """
    Full data validator combining all individual validators.

    Usage:
        validator = DataValidator()
        result = validator.validate(samples)
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.validators = [
            SchemaValidator(),
            SpanValidator(),
            EntityTypeValidator(),
            AssertionValidator(),
            PositionConventionValidator(),
        ]
        if strict:
            self.validators.append(CandidateValidator())

    def validate(self, samples: List[Sample]) -> ValidationResult:
        """Validate all samples."""
        result = ValidationResult(is_valid=True)

        for sample in samples:
            for validator in self.validators:
                sample_result = validator.validate(sample)
                result.is_valid = result.is_valid and sample_result.is_valid
                result.errors.extend(sample_result.errors)
                result.warnings.extend(sample_result.warnings)

        return result

    def validate_file(self, path: str) -> ValidationResult:
        """Validate samples from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        samples = [Sample.from_dict(d) for d in data]
        return self.validate(samples)

    def validate_and_report(self, samples: List[Sample]) -> Dict[str, Any]:
        """Validate and return detailed report."""
        result = self.validate(samples)

        # Group errors by type
        errors_by_field: Dict[str, List[str]] = {}
        for error in result.errors:
            if error.field not in errors_by_field:
                errors_by_field[error.field] = []
            errors_by_field[error.field].append(f"[{error.sample_id}] {error.message}")

        # Group warnings by type
        warnings_by_field: Dict[str, List[str]] = {}
        for warning in result.warnings:
            if warning.field not in warnings_by_field:
                warnings_by_field[warning.field] = []
            warnings_by_field[warning.field].append(f"[{warning.sample_id}] {warning.message}")

        return {
            "is_valid": result.is_valid,
            "total_samples": len(samples),
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "errors_by_field": errors_by_field,
            "warnings_by_field": warnings_by_field,
            "all_errors": [e.message for e in result.errors],
            "all_warnings": [w.message for w in result.warnings],
        }


# =============================================================================
# Linking Sample Validators
# =============================================================================

class ICDLinkingValidator:
    """Validate ICD-10 linking samples."""

    def validate(self, samples: List[Dict[str, Any]]) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        for i, sample in enumerate(samples):
            sample_id = sample.get("id", f"sample_{i}")

            # Required fields
            if "query_text" not in sample:
                result.add_error(sample_id, "query_text", "query_text is required")

            if "mention" not in sample:
                result.add_error(sample_id, "mention", "mention is required")

            if "positive_code" not in sample:
                result.add_error(sample_id, "positive_code", "positive_code is required")

            # Check ICD-10 format
            if "positive_code" in sample:
                code = sample["positive_code"]
                pattern = r'^[A-Z]\d{2}(\.\d+)?$'
                if not re.match(pattern, code):
                    result.add_error(
                        sample_id,
                        "positive_code",
                        f"Invalid ICD-10 format: {code}"
                    )

            # Check negative codes format
            if "negative_codes" in sample:
                for j, neg_code in enumerate(sample["negative_codes"]):
                    if not re.match(pattern, neg_code):
                        result.add_error(
                            sample_id,
                            f"negative_codes[{j}]",
                            f"Invalid ICD-10 format: {neg_code}"
                        )

            # Check mention is in query_text
            if "query_text" in sample and "mention" in sample:
                if sample["mention"] not in sample["query_text"]:
                    result.add_error(
                        sample_id,
                        "mention",
                        f"Mention '{sample['mention']}' not found in query_text"
                    )

        return result


class RxNormLinkingValidator:
    """Validate RxNorm linking samples."""

    def validate(self, samples: List[Dict[str, Any]]) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        for i, sample in enumerate(samples):
            sample_id = sample.get("id", f"sample_{i}")

            # Required fields
            if "query_text" not in sample:
                result.add_error(sample_id, "query_text", "query_text is required")

            if "mention" not in sample:
                result.add_error(sample_id, "mention", "mention is required")

            if "positive_rxcui" not in sample:
                result.add_error(sample_id, "positive_rxcui", "positive_rxcui is required")

            if "positive_name" not in sample:
                result.add_error(sample_id, "positive_name", "positive_name is required")

            # Check RxCUI format (numeric)
            if "positive_rxcui" in sample:
                rxcui = sample["positive_rxcui"]
                if not rxcui.isdigit():
                    result.add_error(
                        sample_id,
                        "positive_rxcui",
                        f"Invalid RxCUI format: {rxcui}"
                    )

            # Check negative rxcuis format
            if "negative_rxcuis" in sample:
                for j, neg_rxcui in enumerate(sample["negative_rxcuis"]):
                    if not neg_rxcui.isdigit():
                        result.add_error(
                            sample_id,
                            f"negative_rxcuis[{j}]",
                            f"Invalid RxCUI format: {neg_rxcui}"
                        )

            # Check mention is in query_text
            if "query_text" in sample and "mention" in sample:
                if sample["mention"] not in sample["query_text"]:
                    result.add_error(
                        sample_id,
                        "mention",
                        f"Mention '{sample['mention']}' not found in query_text"
                    )

        return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for data validator."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate synthetic data")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--strict", "-s", action="store_true", help="Enable strict validation")
    args = parser.parse_args()

    validator = DataValidator(strict=args.strict)
    result = validator.validate_file(args.input)

    print("=" * 60)
    print("Validation Report")
    print("=" * 60)

    if result.is_valid:
        print("\n✅ Validation PASSED")
    else:
        print("\n❌ Validation FAILED")

    print(f"\nErrors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors[:20]:  # Show first 20
            print(f"  - [{error.sample_id}] {error.field}: {error.message}")
        if len(result.errors) > 20:
            print(f"  ... and {len(result.errors) - 20} more")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings[:10]:
            print(f"  - [{warning.sample_id}] {warning.field}: {warning.message}")
        if len(result.warnings) > 10:
            print(f"  ... and {len(result.warnings) - 10} more")


if __name__ == "__main__":
    main()
