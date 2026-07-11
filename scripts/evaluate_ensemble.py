"""
Ensemble Evaluation Script

Compare different extraction strategies.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.entity.confidence import ConfidenceConfig, EntitySource
from src.entity.conflict_logger import ConflictLogger
from src.entity.resolver import EntityResolver
from src.entity.ensemble import EntityEnsemble, SimpleEnsemble
from src.entity.metrics import compute_entity_metrics, print_metrics_report


@dataclass
class EvaluationResult:
    """Result of one evaluation run."""
    name: str
    precision: float
    recall: float
    f1: float
    per_class_f1: Dict[str, Dict[str, float]]
    num_true: int
    num_pred: int
    num_correct: int


def load_test_data(path: str) -> List[Dict[str, Any]]:
    """Load test data from JSONL."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def evaluate_strategy(
    name: str,
    extractor_fn,
    test_samples: List[Dict[str, Any]],
    config: ConfidenceConfig,
) -> EvaluationResult:
    """Evaluate a single extraction strategy.

    Args:
        name: Strategy name
        extractor_fn: Function to extract entities
        test_samples: Test samples with ground truth
        config: Confidence config

    Returns:
        Evaluation result
    """
    all_true = []
    all_pred = []

    for sample in test_samples:
        text = sample.get("text", "")
        true_entities = sample.get("entities", [])

        # Extract using strategy
        pred_entities = extractor_fn(text)

        all_true.extend(true_entities)
        all_pred.extend(pred_entities)

    # Compute metrics
    metrics = compute_entity_metrics(all_true, all_pred)

    return EvaluationResult(
        name=name,
        precision=metrics.get("precision", 0),
        recall=metrics.get("recall", 0),
        f1=metrics.get("entity_f1", 0),
        per_class_f1=metrics.get("per_class_f1", {}),
        num_true=metrics.get("num_true", 0),
        num_pred=metrics.get("num_pred", 0),
        num_correct=metrics.get("num_correct", 0),
    )


def create_regex_only_extractor(config: ConfidenceConfig):
    """Create regex-only extractor."""
    from src.entity.lab_extractor import LabExtractor
    from src.entity.drug_extractor import DrugExtractor

    lab_ext = LabExtractor()
    drug_ext = DrugExtractor()

    def extract(text: str) -> List[Dict[str, Any]]:
        entities = []

        for match in lab_ext.extract_lab_tests(text):
            entities.append({
                "text": match.get("text", ""),
                "start": match.get("start", 0),
                "end": match.get("end", 0),
                "type": "TÊN_XÉT_NGHIỆM",
                "confidence": config.regex_confidence,
                "source": "regex",
            })

        for match in lab_ext.extract_lab_results(text):
            entities.append({
                "text": match.get("text", ""),
                "start": match.get("start", 0),
                "end": match.get("end", 0),
                "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                "confidence": config.regex_confidence,
                "source": "regex",
            })

        for match in drug_ext.extract_drugs(text):
            entities.append({
                "text": match.get("text", ""),
                "start": match.get("start", 0),
                "end": match.get("end", 0),
                "type": "THUỐC",
                "confidence": config.regex_confidence,
                "source": "regex",
            })

        return entities

    return extract


def create_dict_only_extractor(config: ConfidenceConfig):
    """Create dictionary-only extractor."""
    # Simplified - would load actual dictionary
    def extract(text: str) -> List[Dict[str, Any]]:
        # Placeholder - implement with actual dictionary
        return []

    return extract


def create_ner_only_extractor(config: ConfidenceConfig):
    """Create NER-only extractor."""
    # Placeholder - would load actual model
    def extract(text: str) -> List[Dict[str, Any]]:
        return []

    return extract


def create_union_extractor(config: ConfidenceConfig):
    """Create simple union extractor."""
    regex_ext = create_regex_only_extractor(config)

    def extract(text: str) -> List[Dict[str, Any]]:
        entities = regex_ext(text)
        # Simple deduplicate by span
        seen = set()
        unique = []
        for e in entities:
            span = (e["start"], e["end"], e["type"])
            if span not in seen:
                seen.add(span)
                unique.append(e)
        return unique

    return extract


def create_ensemble_extractor(config: ConfidenceConfig):
    """Create full ensemble extractor."""
    ensemble = EntityEnsemble(config=config)
    ensemble.set_extractors(
        regex_extractor=create_regex_only_extractor(config),
    )
    return lambda text: ensemble.extract(text)


def print_comparison(results: List[EvaluationResult]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("ENSEMBLE EVALUATION COMPARISON")
    print("=" * 80)
    print(f"\n{'Strategy':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(f"{r.name:<25} {r.precision:>10.4f} {r.recall:>10.4f} {r.f1:>10.4f} "
              f"{r.num_correct:>10} {r.num_true:>10}")

    print("=" * 80)

    # Best strategy
    best = max(results, key=lambda x: x.f1)
    print(f"\nBest Strategy: {best.name} (F1={best.f1:.4f})")

    # Per-class breakdown for best
    print(f"\nPer-Class F1 for {best.name}:")
    for entity_type, metrics in best.per_class_f1.items():
        print(f"  {entity_type:<30}: F1={metrics['f1']:.4f} (support={metrics.get('support', 0)})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble strategies")
    parser.add_argument("--data", type=str, default="data/processed/test.jsonl",
                        help="Test data path")
    parser.add_argument("--output", type=str, default="outputs/ensemble_eval.json",
                        help="Output path")
    args = parser.parse_args()

    # Load test data
    test_samples = load_test_data(args.data)
    print(f"Loaded {len(test_samples)} test samples")

    # Config
    config = ConfidenceConfig()

    # Run evaluations
    results = []

    # 1. Regex only
    print("\nEvaluating: Regex Only")
    results.append(evaluate_strategy(
        "regex_only",
        create_regex_only_extractor(config),
        test_samples,
        config,
    ))

    # 2. Dictionary only (if implemented)
    print("Evaluating: Dictionary Only")
    results.append(evaluate_strategy(
        "dictionary_only",
        create_dict_only_extractor(config),
        test_samples,
        config,
    ))

    # 3. NER only (if model available)
    print("Evaluating: NER Only")
    results.append(evaluate_strategy(
        "ner_only",
        create_ner_only_extractor(config),
        test_samples,
        config,
    ))

    # 4. Simple union
    print("Evaluating: Simple Union")
    results.append(evaluate_strategy(
        "simple_union",
        create_union_extractor(config),
        test_samples,
        config,
    ))

    # 5. Full ensemble
    print("Evaluating: Full Ensemble")
    results.append(evaluate_strategy(
        "full_ensemble",
        create_ensemble_extractor(config),
        test_samples,
        config,
    ))

    # Print comparison
    print_comparison(results)

    # Save results
    output = {
        "results": [
            {
                "name": r.name,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "per_class_f1": r.per_class_f1,
                "num_true": r.num_true,
                "num_pred": r.num_pred,
                "num_correct": r.num_correct,
            }
            for r in results
        ],
        "best_strategy": max(results, key=lambda x: x.f1).name,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
