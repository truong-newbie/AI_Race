"""
Validate Dataset

Script to validate synthetic datasets against all rules:
- Schema validation
- Span alignment
- Entity type validation
- Assertion validation
- Duplicate detection
- Cross-dataset validation (train vs validation)

Usage:
    python scripts/validate_dataset.py --input data/synthetic/template_samples.jsonl
    python scripts/validate_dataset.py --input data/processed/train.jsonl --strict
    python scripts/validate_dataset.py --dir data/synthetic
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.validators import DataValidator, ICDLinkingValidator, RxNormLinkingValidator
from src.data.schema import load_samples, Sample


def validate_standard_samples(path: str, strict: bool = False) -> Dict[str, Any]:
    """Validate standard sample files."""
    validator = DataValidator(strict=strict)

    try:
        result = validator.validate_and_report(load_samples(path))
        return result
    except Exception as e:
        return {"error": str(e), "file": path}


def validate_linking_samples(path: str, link_type: str = "icd") -> Dict[str, Any]:
    """Validate linking sample files."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if link_type == "icd":
        validator = ICDLinkingValidator()
    else:
        validator = RxNormLinkingValidator()

    result = validator.validate(samples)
    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "errors": [{"field": e.field, "message": e.message} for e in result.errors],
        "warnings": [{"field": w.field, "message": w.message} for w in result.warnings],
    }


def check_span_alignment(path: str) -> Dict[str, Any]:
    """Check if all spans align with text."""
    samples = load_samples(path)

    errors = []
    total_checked = 0

    for sample in samples:
        for i, entity in enumerate(sample.entities):
            total_checked += 1
            try:
                extracted = sample.text[entity.start:entity.end]
                if extracted != entity.text:
                    errors.append({
                        "sample_id": sample.id,
                        "entity_index": i,
                        "entity_text": entity.text,
                        "extracted": extracted,
                        "start": entity.start,
                        "end": entity.end,
                    })
            except IndexError:
                errors.append({
                    "sample_id": sample.id,
                    "entity_index": i,
                    "error": "IndexError in slice",
                    "start": entity.start,
                    "end": entity.end,
                    "text_length": len(sample.text),
                })

    return {
        "total_checked": total_checked,
        "alignment_errors": len(errors),
        "errors": errors[:50],  # Limit to first 50
    }


def check_cross_dataset_leakage(train_path: str, val_path: str) -> Dict[str, Any]:
    """Check for text leakage between train and validation sets."""
    train_samples = load_samples(train_path)
    val_samples = load_samples(val_path)

    train_texts = {s.text for s in train_samples}
    leakage = []

    for sample in val_samples:
        if sample.text in train_texts:
            leakage.append(sample.id)

    return {
        "train_count": len(train_samples),
        "validation_count": len(val_samples),
        "leakage_count": len(leakage),
        "leakage_ids": leakage[:20],
    }


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic datasets")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--dir", "-d", type=str, help="Directory containing JSONL files")
    parser.add_argument("--strict", "-s", action="store_true", help="Enable strict validation")
    parser.add_argument("--check-leakage", action="store_true", help="Check train/validation leakage")
    parser.add_argument("--train", type=str, default="data/processed/train.jsonl", help="Train file for leakage check")
    parser.add_argument("--validation", type=str, default="data/validation/manual_validation_template.jsonl", help="Validation file for leakage check")
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    if args.input:
        # Validate single file
        print(f"\nValidating: {args.input}")

        if args.input.endswith("linking_samples.jsonl"):
            link_type = "icd" if "icd" in args.input else "rxnorm"
            result = validate_linking_samples(args.input, link_type)
        else:
            result = validate_standard_samples(args.input, args.strict)

        print_result(result, args.input)

    elif args.dir:
        # Validate all files in directory
        print(f"\nValidating directory: {args.dir}")

        results = {}
        for filename in os.listdir(args.dir):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(args.dir, filename)
                print(f"\n  {filename}...")

                if "linking" in filename:
                    link_type = "icd" if "icd" in filename else "rxnorm"
                    result = validate_linking_samples(filepath, link_type)
                else:
                    result = validate_standard_samples(filepath, args.strict)

                results[filename] = result
                error_count = result.get('error_count', 'N/A')
                print(f"    Errors: {error_count}")

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for filename, result in results.items():
            status = "[OK]" if (result.get("is_valid") or result.get("error_count", 1) == 0) else "[FAIL]"
            print(f"  {status} {filename}")

    else:
        print("Error: Specify --input or --dir")
        return

    # Check leakage if requested
    if args.check_leakage:
        print("\n" + "=" * 60)
        print("Checking Train/Validation Leakage")
        print("=" * 60)

        if os.path.exists(args.train) and os.path.exists(args.validation):
            result = check_cross_dataset_leakage(args.train, args.validation)
            print(f"\nTrain: {result['train_count']} samples")
            print(f"Validation: {result['validation_count']} samples")
            print(f"Leakage: {result['leakage_count']} samples")

            if result['leakage_count'] > 0:
                print("\n[WARNING] Text leakage detected!")
                print("Samples in both train and validation:")
                for lid in result['leakage_ids']:
                    print(f"  - {lid}")
            else:
                print("\n[OK] No leakage detected")
        else:
            print(f"\nSkipping leakage check: train or validation file not found")


def print_result(result: Dict[str, Any], filename: str):
    """Print validation result."""
    print("\n" + "-" * 40)

    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return

    if result.get("is_valid"):
        print("[OK] Validation PASSED")
    else:
        print("[FAIL] Validation FAILED")

    print(f"\nTotal samples: {result.get('total_samples', 'N/A')}")
    print(f"Errors: {result.get('error_count', result.get('errors', {}).get('error_count', len(result.get('errors', []))))}")
    print(f"Warnings: {result.get('warning_count', result.get('warnings', {}).get('warning_count', len(result.get('warnings', []))))}")

    # Print errors
    errors = result.get("errors", result.get("errors_by_field", []))
    if isinstance(errors, dict):
        errors = [e for es in errors.values() for e in es]
    if errors:
        print("\nErrors:")
        for error in errors[:10]:
            if isinstance(error, dict):
                print(f"  - [{error.get('sample_id', 'N/A')}] {error.get('message', error)}")
            else:
                print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
