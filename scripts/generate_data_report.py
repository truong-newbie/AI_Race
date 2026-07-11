"""
Generate Data Report

Script to generate comprehensive reports for datasets:
- Total records and entities
- Entity distribution by type
- Assertion distribution
- ICD-10 and RxNorm coverage
- Train/dev/test split statistics
- Duplicate detection results
- Text length distribution
- Entity count per sample

Usage:
    python scripts/generate_data_report.py --input data/synthetic/template_samples.jsonl
    python scripts/generate_data_report.py --train data/processed/train.jsonl --dev data/processed/dev.jsonl --test data/processed/internal_test.jsonl
    python scripts/generate_data_report.py --dir data/synthetic --output report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.report import ReportGenerator
from src.data.schema import load_samples


def generate_single_report(input_path: str, output_path: str = None, format: str = "json") -> None:
    """Generate report for a single file."""
    generator = ReportGenerator()
    report = generator.generate_from_file(input_path)

    if format == "markdown":
        output = generator.format_markdown(report)
    else:
        output = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report saved to {output_path}")
    else:
        print(output)


def generate_split_report(train_path: str, dev_path: str, test_path: str,
                         output_path: str = None) -> None:
    """Generate report for split datasets."""
    generator = ReportGenerator()

    print("Loading datasets...")
    train_samples = load_samples(train_path)
    dev_samples = load_samples(dev_path)
    test_samples = load_samples(test_path)

    report = generator.generate_split_report(train_samples, dev_samples, test_samples)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {output_path}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


def generate_directory_report(directory: str, output_path: str = None) -> None:
    """Generate reports for all files in a directory."""
    generator = ReportGenerator()

    files = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            files[filename] = filepath

    print(f"Found {len(files)} JSONL files")

    reports = {}
    for name, path in files.items():
        try:
            report = generator.generate_from_file(path)
            reports[name] = report.to_dict()
        except Exception as e:
            reports[name] = {"error": str(e)}

    output = {
        "directory": directory,
        "files": list(files.keys()),
        "reports": reports,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {output_path}")
    else:
        # Print summary only (not full JSON) for console compatibility
        print(f"Directory: {output.get('directory', 'N/A')}")
        print(f"Files: {output.get('files', [])}")
        print(f"Reports generated: {len(output.get('reports', {}))}")


def main():
    parser = argparse.ArgumentParser(description="Generate data reports")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output file")
    parser.add_argument("--format", "-f", choices=["json", "markdown"], default="json", help="Output format")
    parser.add_argument("--train", type=str, help="Train JSONL file")
    parser.add_argument("--dev", type=str, help="Dev JSONL file")
    parser.add_argument("--test", type=str, help="Test JSONL file")
    parser.add_argument("--dir", "-d", type=str, help="Directory containing JSONL files")
    args = parser.parse_args()

    print("=" * 60)
    print("Data Report Generator")
    print("=" * 60)

    if args.dir:
        print(f"\nGenerating reports for directory: {args.dir}")
        generate_directory_report(args.dir, args.output)

    elif args.train and args.dev and args.test:
        print(f"\nGenerating split report...")
        print(f"  Train: {args.train}")
        print(f"  Dev: {args.dev}")
        print(f"  Test: {args.test}")
        generate_split_report(args.train, args.dev, args.test, args.output)

    elif args.input:
        print(f"\nGenerating report for: {args.input}")
        generate_single_report(args.input, args.output, args.format)

    else:
        print("Error: Specify --input, --train/--dev/--test, or --dir")
        return

    print("\n[OK] Report generation complete!")


if __name__ == "__main__":
    main()
