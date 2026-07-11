"""
Generate Synthetic Medical Data

Main script to generate all synthetic data:
1. Template samples
2. Data variants
3. ICD-10 linking samples
4. RxNorm linking samples
5. Validation templates

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --seed 42
    python scripts/generate_synthetic_data.py --skip-variants
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.template_generator import TemplateGenerator
from src.data.variant_generator import VariantGenerator, VariantConfig
from src.data.linking_generator import LinkingGenerator
from src.data.validation_generator import ValidationTemplateGenerator
from src.data.schema import save_jsonl, load_jsonl, Sample


def ensure_dirs():
    """Ensure output directories exist."""
    dirs = [
        "data/synthetic",
        "data/validation",
        "data/processed",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def generate_templates(seed: int, count: int) -> int:
    """Generate template samples."""
    print("Generating template samples...")
    generator = TemplateGenerator(seed=seed)
    samples = generator.generate_all(count=count)

    output_path = "data/synthetic/template_samples.jsonl"
    save_jsonl(output_path, [s.to_dict() for s in samples])
    print(f"  Generated {len(samples)} template samples -> {output_path}")
    return len(samples)


def generate_variants(seed: int, max_variants: int) -> int:
    """Generate data variants."""
    print("Generating data variants...")

    # Load template samples
    input_path = "data/synthetic/template_samples.jsonl"
    if not os.path.exists(input_path):
        print(f"  Warning: {input_path} not found, skipping variants")
        return 0

    samples_data = load_jsonl(input_path)
    samples = [Sample.from_dict(d) for d in samples_data]

    # Generate variants
    config = VariantConfig(
        case_variations=True,
        diacritic_variations=True,
        typo_variations=True,
        punctuation_variations=True,
        max_variants_per_sample=max_variants,
    )

    generator = VariantGenerator(config=config, seed=seed)
    variants = generator.generate_variants_batch(samples)

    output_path = "data/synthetic/variant_samples.jsonl"
    save_jsonl(output_path, [v.to_dict() for v in variants])
    print(f"  Generated {len(variants)} variants -> {output_path}")
    return len(variants)


def generate_icd_linking(seed: int, count: int) -> int:
    """Generate ICD-10 linking samples."""
    print("Generating ICD-10 linking samples...")
    generator = LinkingGenerator(seed=seed)
    samples = generator.generate_icd10_samples(count)

    output_path = "data/synthetic/icd_linking_samples.jsonl"
    save_jsonl(output_path, [s.to_dict() for s in samples])
    print(f"  Generated {len(samples)} ICD-10 linking samples -> {output_path}")
    return len(samples)


def generate_rxnorm_linking(seed: int, count: int) -> int:
    """Generate RxNorm linking samples."""
    print("Generating RxNorm linking samples...")
    generator = LinkingGenerator(seed=seed)
    samples = generator.generate_rxnorm_samples(count)

    output_path = "data/synthetic/rxnorm_linking_samples.jsonl"
    save_jsonl(output_path, [s.to_dict() for s in samples])
    print(f"  Generated {len(samples)} RxNorm linking samples -> {output_path}")
    return len(samples)


def generate_validation_templates(seed: int) -> int:
    """Generate manual validation templates."""
    print("Generating validation templates...")
    generator = ValidationTemplateGenerator(seed=seed)
    templates = generator.generate_all()

    output_path = "data/validation/manual_validation_template.jsonl"
    save_jsonl(output_path, [t.to_dict() for t in templates])

    # Print summary
    summary = generator.generate_summary(templates)
    print(f"  Generated {summary['total_templates']} validation templates -> {output_path}")
    print(f"    By category: {summary['by_category']}")
    print(f"    By difficulty: {summary['by_difficulty']}")
    return summary['total_templates']


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--template-count", type=int, default=100, help="Number of template samples")
    parser.add_argument("--max-variants", type=int, default=3, help="Max variants per sample")
    parser.add_argument("--icd-count", type=int, default=100, help="Number of ICD-10 linking samples")
    parser.add_argument("--rx-count", type=int, default=100, help="Number of RxNorm linking samples")
    parser.add_argument("--skip-variants", action="store_true", help="Skip variant generation")
    parser.add_argument("--skip-linking", action="store_true", help="Skip linking generation")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation template generation")
    args = parser.parse_args()

    print("=" * 60)
    print("Synthetic Medical Data Generation")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print()

    ensure_dirs()

    total_samples = 0

    # Generate templates
    total_samples += generate_templates(args.seed, args.template_count)

    # Generate variants
    if not args.skip_variants:
        total_samples += generate_variants(args.seed, args.max_variants)

    # Generate linking samples
    if not args.skip_linking:
        total_samples += generate_icd_linking(args.seed, args.icd_count)
        total_samples += generate_rxnorm_linking(args.seed, args.rx_count)

    # Generate validation templates
    if not args.skip_validation:
        total_samples += generate_validation_templates(args.seed)

    print()
    print("=" * 60)
    print(f"Generation complete! Total samples: {total_samples}")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  data/synthetic/template_samples.jsonl")
    if not args.skip_variants:
        print("  data/synthetic/variant_samples.jsonl")
    if not args.skip_linking:
        print("  data/synthetic/icd_linking_samples.jsonl")
        print("  data/synthetic/rxnorm_linking_samples.jsonl")
    if not args.skip_validation:
        print("  data/validation/manual_validation_template.jsonl")


if __name__ == "__main__":
    main()
