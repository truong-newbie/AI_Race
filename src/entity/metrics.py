"""
NER Metrics Module

Entity-level precision, recall, F1 metrics.
"""

from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def compute_entity_metrics(
    true_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute entity-level metrics.

    Args:
        true_entities: List of true entity dicts
        pred_entities: List of predicted entity dicts

    Returns:
        Dict with precision, recall, f1, per_class_f1, boundary_errors, wrong_type_errors
    """
    # Convert to sets for comparison
    true_set = set(_entity_to_tuple(e) for e in true_entities)
    pred_set = set(_entity_to_tuple(e) for e in pred_entities)

    # Basic counts
    num_true = len(true_set)
    num_pred = len(pred_set)
    num_correct = len(true_set & pred_set)

    # Precision, Recall, F1
    precision = num_correct / num_pred if num_pred > 0 else 0.0
    recall = num_correct / num_true if num_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-class F1
    per_class_f1 = compute_per_class_f1(true_entities, pred_entities)

    # Error analysis
    boundary_errors, wrong_type_errors = analyze_errors(true_entities, pred_entities)

    return {
        "precision": precision,
        "recall": recall,
        "entity_f1": f1,
        "num_true": num_true,
        "num_pred": num_pred,
        "num_correct": num_correct,
        "per_class_f1": per_class_f1,
        "boundary_errors": boundary_errors,
        "wrong_type_errors": wrong_type_errors,
    }


def _entity_to_tuple(entity: Dict[str, Any]) -> Tuple:
    """Convert entity dict to tuple for set comparison."""
    return (entity["text"], entity["start"], entity["end"], entity["type"])


def compute_per_class_f1(
    true_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, F1.

    Args:
        true_entities: List of true entities
        pred_entities: List of predicted entities

    Returns:
        Dict mapping entity type to precision, recall, f1
    """
    from src.entity.labels import NER_ENTITY_TYPES

    results = {}

    for entity_type in NER_ENTITY_TYPES:
        # Filter by type
        true_type = [e for e in true_entities if e["type"] == entity_type]
        pred_type = [e for e in pred_entities if e["type"] == entity_type]

        true_set = set(_entity_to_tuple(e) for e in true_type)
        pred_set = set(_entity_to_tuple(e) for e in pred_type)

        num_true = len(true_set)
        num_pred = len(pred_set)
        num_correct = len(true_set & pred_set)

        precision = num_correct / num_pred if num_pred > 0 else 0.0
        recall = num_correct / num_true if num_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": num_true,
        }

    return results


def analyze_errors(
    true_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """Analyze prediction errors.

    Args:
        true_entities: List of true entities
        pred_entities: List of predicted entities

    Returns:
        Tuple of (boundary_errors, wrong_type_errors)
    """
    true_set = set(_entity_to_tuple(e) for e in true_entities)
    pred_set = set(_entity_to_tuple(e) for e in pred_entities)

    boundary_errors = 0
    wrong_type_errors = 0

    # Find false positives (predictions not in truth)
    for pred in pred_set:
        if pred not in true_set:
            # Check if same boundaries exist with different type
            text, start, end, pred_type = pred

            # Find matching boundary with different type
            matching = [t for t in true_set if t[0] == text and t[1] == start and t[2] == end]

            if matching:
                # Same boundaries, wrong type
                wrong_type_errors += 1
            else:
                # Wrong boundaries (partial overlap)
                boundary_errors += 1

    return boundary_errors, wrong_type_errors


def detailed_error_analysis(
    true_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Detailed error analysis with examples.

    Args:
        true_entities: List of true entities
        pred_entities: List of predicted entities

    Returns:
        Dict with detailed error information
    """
    true_set = set(_entity_to_tuple(e) for e in true_entities)
    pred_set = set(_entity_to_tuple(e) for e in pred_entities)

    # False negatives (missed entities)
    false_negatives = []
    for true in true_set:
        if true not in pred_set:
            false_negatives.append({
                "text": true[0],
                "start": true[1],
                "end": true[2],
                "type": true[3],
            })

    # False positives (spurious entities)
    false_positives = []
    for pred in pred_set:
        if pred not in true_set:
            # Check wrong type vs boundary error
            text, start, end, pred_type = pred
            matching_true = [t for t in true_set if t[0] == text and t[1] == start and t[2] == end]

            if matching_true:
                true_type = matching_true[0][3]
                error_type = "wrong_type"
                error_detail = f"predicted {pred_type}, actual {true_type}"
            else:
                error_type = "boundary_error"
                # Find partial overlap
                partial = [t for t in true_set if t[0] == text]
                if partial:
                    error_detail = f"predicted ({start}-{end}), actual boundaries: {[(t[1], t[2]) for t in partial]}"
                else:
                    error_detail = "no matching text"

            false_positives.append({
                "text": text,
                "start": start,
                "end": end,
                "type": pred_type,
                "error_type": error_type,
                "detail": error_detail,
            })

    # Correct predictions
    correct = list(true_set & pred_set)

    return {
        "num_correct": len(correct),
        "num_false_negatives": len(false_negatives),
        "num_false_positives": len(false_positives),
        "false_negatives": false_negatives[:10],  # Limit examples
        "false_positives": false_positives[:10],
    }


def print_metrics_report(metrics: Dict[str, Any]) -> str:
    """Format metrics as a readable report.

    Args:
        metrics: Metrics dict from compute_entity_metrics

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("NER Evaluation Report")
    lines.append("=" * 60)

    # Overall metrics
    lines.append("\nOverall Metrics:")
    lines.append(f"  Precision: {metrics['precision']:.4f}")
    lines.append(f"  Recall:    {metrics['recall']:.4f}")
    lines.append(f"  F1:        {metrics['entity_f1']:.4f}")
    lines.append(f"  Correct:   {metrics['num_correct']}/{metrics['num_true']}")

    # Per-class metrics
    lines.append("\nPer-Class F1:")
    per_class = metrics.get("per_class_f1", {})
    for entity_type, class_metrics in per_class.items():
        f1 = class_metrics["f1"]
        support = class_metrics["support"]
        lines.append(f"  {entity_type:30s}: F1={f1:.4f} (n={support})")

    # Error analysis
    lines.append("\nError Analysis:")
    lines.append(f"  Boundary errors: {metrics.get('boundary_errors', 0)}")
    lines.append(f"  Wrong type errors: {metrics.get('wrong_type_errors', 0)}")

    lines.append("=" * 60)

    return "\n".join(lines)


# =============================================================================
# Test Functions
# =============================================================================

def test_basic_metrics():
    """Test basic metrics computation."""
    true_entities = [
        {"text": "viêm phổi", "start": 17, "end": 27, "type": "CHẨN_ĐOÁN"},
        {"text": "Paracetamol", "start": 30, "end": 41, "type": "THUỐC"},
    ]

    pred_entities = [
        {"text": "viêm phổi", "start": 17, "end": 27, "type": "CHẨN_ĐOÁN"},
    ]

    metrics = compute_entity_metrics(true_entities, pred_entities)

    assert metrics["num_true"] == 2
    assert metrics["num_pred"] == 1
    assert metrics["num_correct"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    print(f"[OK] Basic metrics test passed: F1={metrics['entity_f1']:.4f}")


def test_perfect_match():
    """Test perfect match."""
    true_entities = [
        {"text": "viêm phổi", "start": 17, "end": 27, "type": "CHẨN_ĐOÁN"},
    ]

    pred_entities = [
        {"text": "viêm phổi", "start": 17, "end": 27, "type": "CHẨN_ĐOÁN"},
    ]

    metrics = compute_entity_metrics(true_entities, pred_entities)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["entity_f1"] == 1.0
    print(f"[OK] Perfect match test passed: F1={metrics['entity_f1']:.4f}")


def test_wrong_type():
    """Test wrong type error detection."""
    true_entities = [
        {"text": "viêm phổi", "start": 17, "end": 27, "type": "CHẨN_ĐOÁN"},
    ]

    pred_entities = [
        {"text": "viêm phổi", "start": 17, "end": 27, "type": "TRIỆU_CHỨNG"},
    ]

    metrics = compute_entity_metrics(true_entities, pred_entities)

    assert metrics["wrong_type_errors"] == 1
    assert metrics["boundary_errors"] == 0
    print(f"[OK] Wrong type test passed: {metrics['wrong_type_errors']} wrong type errors")


def test_boundary_error():
    """Test boundary error detection."""
    true_entities = [
        {"text": "viêm phổi", "start": 17, "end": 27, "type": "CHẨN_ĐOÁN"},
    ]

    # Predicted with slightly wrong boundaries
    pred_entities = [
        {"text": "viêm phổ", "start": 17, "end": 25, "type": "CHẨN_ĐOÁN"},
    ]

    metrics = compute_entity_metrics(true_entities, pred_entities)

    assert metrics["boundary_errors"] == 1
    print(f"[OK] Boundary error test passed: {metrics['boundary_errors']} boundary errors")


if __name__ == "__main__":
    print("Running metrics tests...")
    test_basic_metrics()
    test_perfect_match()
    test_wrong_type()
    test_boundary_error()
    print("All metrics tests passed!")
