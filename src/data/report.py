"""
Data Report Generator

Generate comprehensive reports for synthetic datasets:
- Total records and entities
- Entity distribution by type
- Assertion distribution
- ICD-10 and RxNorm coverage
- Train/dev/test split statistics
- Duplicate detection results
- Text length distribution
- Entity count per sample
"""

import json
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics

from .schema import Sample, Entity, load_samples


@dataclass
class DataReport:
    """Comprehensive data report."""
    total_samples: int = 0
    total_entities: int = 0
    entity_counts_by_type: Dict[str, int] = field(default_factory=dict)
    assertion_counts: Dict[str, int] = field(default_factory=dict)
    icd_codes: Set[str] = field(default_factory=set)
    rxcuis: Set[str] = field(default_factory=set)
    samples_by_source: Dict[str, int] = field(default_factory=dict)
    text_length_stats: Dict[str, float] = field(default_factory=dict)
    entities_per_sample_stats: Dict[str, float] = field(default_factory=dict)
    validation_errors: int = 0
    duplicates_removed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "total_entities": self.total_entities,
            "entity_counts_by_type": self.entity_counts_by_type,
            "assertion_counts": self.assertion_counts,
            "unique_icd_codes": len(self.icd_codes),
            "unique_rxcuis": len(self.rxcuis),
            "samples_by_source": self.samples_by_source,
            "text_length_stats": self.text_length_stats,
            "entities_per_sample_stats": self.entities_per_sample_stats,
            "validation_errors": self.validation_errors,
            "duplicates_removed": self.duplicates_removed,
        }


class ReportGenerator:
    """
    Generate reports for datasets.

    Usage:
        generator = ReportGenerator()
        report = generator.generate(samples)
        print(report.to_dict())
    """

    def __init__(self):
        pass

    def generate(self, samples: List[Sample]) -> DataReport:
        """Generate report from samples."""
        report = DataReport()

        # Basic counts
        report.total_samples = len(samples)
        text_lengths = []
        entities_per_sample = []

        # Entity tracking
        entity_types: Counter = Counter()
        assertion_counts: Counter = Counter()
        icd_codes: Set[str] = set()
        rxcuis: Set[str] = set()

        # Source tracking
        samples_by_source: Counter = Counter()

        for sample in samples:
            text_lengths.append(len(sample.text))
            entities_per_sample.append(len(sample.entities))
            report.total_entities += len(sample.entities)

            # Track source
            samples_by_source[sample.source] += 1

            for entity in sample.entities:
                # Entity type
                entity_types[entity.type] += 1

                # Assertions
                for assertion in entity.assertions:
                    assertion_counts[assertion] += 1

                # ICD codes (from candidates)
                if entity.type == "CHẨN_ĐOÁN":
                    for candidate in entity.candidates:
                        if self._looks_like_icd10(candidate):
                            icd_codes.add(candidate)

                # RxNorm CUIs (from candidates)
                if entity.type == "THUỐC":
                    for candidate in entity.candidates:
                        if candidate.isdigit():
                            rxcuis.add(candidate)

        report.entity_counts_by_type = dict(entity_types)
        report.assertion_counts = dict(assertion_counts)
        report.icd_codes = icd_codes
        report.rxcuis = rxcuis
        report.samples_by_source = dict(samples_by_source)

        # Text length statistics
        if text_lengths:
            report.text_length_stats = {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "mean": statistics.mean(text_lengths),
                "median": statistics.median(text_lengths),
                "stdev": statistics.stdev(text_lengths) if len(text_lengths) > 1 else 0,
            }

        # Entities per sample statistics
        if entities_per_sample:
            report.entities_per_sample_stats = {
                "min": min(entities_per_sample),
                "max": max(entities_per_sample),
                "mean": statistics.mean(entities_per_sample),
                "median": statistics.median(entities_per_sample),
                "stdev": statistics.stdev(entities_per_sample) if len(entities_per_sample) > 1 else 0,
            }

        return report

    def generate_from_file(self, path: str) -> DataReport:
        """Generate report from JSONL file."""
        samples = load_samples(path)
        return self.generate(samples)

    def generate_split_report(self, train: List[Sample], dev: List[Sample], test: List[Sample]) -> Dict[str, Any]:
        """Generate report for split datasets."""
        train_report = self.generate(train)
        dev_report = self.generate(dev)
        test_report = self.generate(test)

        # Calculate ratios
        total = train_report.total_samples + dev_report.total_samples + test_report.total_samples

        return {
            "overview": {
                "total_samples": total,
                "train_count": train_report.total_samples,
                "train_ratio": train_report.total_samples / total if total > 0 else 0,
                "dev_count": dev_report.total_samples,
                "dev_ratio": dev_report.total_samples / total if total > 0 else 0,
                "test_count": test_report.total_samples,
                "test_ratio": test_report.total_samples / total if total > 0 else 0,
            },
            "train": train_report.to_dict(),
            "dev": dev_report.to_dict(),
            "test": test_report.to_dict(),
        }

    def generate_comparison_report(self, files: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate comparison report across multiple files.

        Args:
            files: Dict mapping name -> file path

        Returns:
            Comparison report
        """
        reports = {}
        for name, path in files.items():
            try:
                reports[name] = self.generate_from_file(path).to_dict()
            except FileNotFoundError:
                reports[name] = {"error": "File not found"}

        return {
            "files": list(files.keys()),
            "reports": reports,
        }

    def _looks_like_icd10(self, code: str) -> bool:
        """Check if code looks like ICD-10 format."""
        import re
        pattern = r'^[A-Z]\d{2}(\.\d+)?$'
        return bool(re.match(pattern, code))

    def format_markdown(self, report: DataReport) -> str:
        """Format report as markdown."""
        lines = [
            "# Data Report",
            "",
            f"## Overview",
            f"- **Total Samples**: {report.total_samples:,}",
            f"- **Total Entities**: {report.total_entities:,}",
            f"- **Unique ICD Codes**: {len(report.icd_codes)}",
            f"- **Unique RxCUI**: {len(report.rxcuis)}",
            "",
            f"## Entity Distribution",
        ]

        for etype, count in sorted(report.entity_counts_by_type.items()):
            pct = count / report.total_entities * 100 if report.total_entities > 0 else 0
            lines.append(f"- **{etype}**: {count:,} ({pct:.1f}%)")

        lines.extend([
            "",
            f"## Assertion Distribution",
        ])

        for assertion, count in sorted(report.assertion_counts.items()):
            lines.append(f"- **{assertion}**: {count:,}")

        lines.extend([
            "",
            f"## Source Distribution",
        ])

        for source, count in sorted(report.samples_by_source.items()):
            pct = count / report.total_samples * 100 if report.total_samples > 0 else 0
            lines.append(f"- **{source}**: {count:,} ({pct:.1f}%)")

        if report.text_length_stats:
            lines.extend([
                "",
                f"## Text Length Statistics",
                f"- Min: {report.text_length_stats.get('min', 0):,}",
                f"- Max: {report.text_length_stats.get('max', 0):,}",
                f"- Mean: {report.text_length_stats.get('mean', 0):.1f}",
                f"- Median: {report.text_length_stats.get('median', 0):.1f}",
            ])

        if report.entities_per_sample_stats:
            lines.extend([
                "",
                f"## Entities per Sample Statistics",
                f"- Min: {report.entities_per_sample_stats.get('min', 0):,}",
                f"- Max: {report.entities_per_sample_stats.get('max', 0):,}",
                f"- Mean: {report.entities_per_sample_stats.get('mean', 0):.1f}",
                f"- Median: {report.entities_per_sample_stats.get('median', 0):.1f}",
            ])

        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for report generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate data reports")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output file (default: stdout)")
    parser.add_argument("--format", "-f", choices=["json", "markdown"], default="json", help="Output format")
    parser.add_argument("--train", type=str, help="Train JSONL file")
    parser.add_argument("--dev", type=str, help="Dev JSONL file")
    parser.add_argument("--test", type=str, help="Test JSONL file")
    args = parser.parse_args()

    generator = ReportGenerator()

    if args.train and args.dev and args.test:
        # Generate split report
        train_samples = load_samples(args.train)
        dev_samples = load_samples(args.dev)
        test_samples = load_samples(args.test)
        report = generator.generate_split_report(train_samples, dev_samples, test_samples)
    elif args.input:
        # Generate single file report
        report = generator.generate_from_file(args.input).to_dict()
    else:
        print("Error: Specify --input or all of --train, --dev, --test")
        return

    # Output
    if args.format == "markdown" and "train" not in report:
        report_obj = generator.generate_from_file(args.input) if args.input else None
        if report_obj:
            output = generator.format_markdown(report_obj)
        else:
            output = json.dumps(report, indent=2, ensure_ascii=False)
    else:
        output = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
