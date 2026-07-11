"""
Tests for NER Metrics

Tests entity-level precision, recall, F1 metrics.
"""

import pytest

from src.entity.metrics import (
    compute_entity_metrics,
    compute_per_class_f1,
    analyze_errors,
    detailed_error_analysis,
    print_metrics_report,
)


class TestEntityMetrics:
    """Test compute_entity_metrics function."""

    def test_perfect_match(self):
        """Test perfect match returns F1=1.0."""
        true_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]
        pred_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]

        metrics = compute_entity_metrics(true_entities, pred_entities)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["entity_f1"] == 1.0
        assert metrics["num_correct"] == 1

    def test_partial_match(self):
        """Test partial match with lower scores."""
        true_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]
        pred_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
            {"text": "ho", "start": 11, "end": 13, "type": "TRIỆU_CHỨNG"},
        ]

        metrics = compute_entity_metrics(true_entities, pred_entities)

        assert metrics["precision"] == 0.5  # 1/2
        assert metrics["recall"] == 1.0    # 1/1
        assert abs(metrics["entity_f1"] - 0.666) < 0.01

    def test_no_match(self):
        """Test no matches."""
        true_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]
        pred_entities = [
            {"text": "ho", "start": 0, "end": 2, "type": "TRIỆU_CHỨNG"},
        ]

        metrics = compute_entity_metrics(true_entities, pred_entities)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["entity_f1"] == 0.0

    def test_empty_predictions(self):
        """Test with no predictions."""
        true_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]
        pred_entities = []

        metrics = compute_entity_metrics(true_entities, pred_entities)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["entity_f1"] == 0.0
        assert metrics["num_pred"] == 0

    def test_empty_truth(self):
        """Test with no true entities."""
        true_entities = []
        pred_entities = [
            {"text": "ho", "start": 0, "end": 2, "type": "TRIỆU_CHỨNG"},
        ]

        metrics = compute_entity_metrics(true_entities, pred_entities)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["entity_f1"] == 0.0
        assert metrics["num_true"] == 0

    def test_both_empty(self):
        """Test with no entities at all."""
        true_entities = []
        pred_entities = []

        metrics = compute_entity_metrics(true_entities, pred_entities)

        # All zeros, no division by zero
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["entity_f1"] == 0.0


class TestPerClassF1:
    """Test per-class F1 computation."""

    def test_per_class_f1_single_class(self):
        """Test per-class F1 for single entity type."""
        true_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
            {"text": "viêm họng", "start": 11, "end": 19, "type": "CHẨN_ĐOÁN"},
        ]
        pred_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]

        per_class = compute_per_class_f1(true_entities, pred_entities)

        assert "CHẨN_ĐOÁN" in per_class
        assert per_class["CHẨN_ĐOÁN"]["support"] == 2
        assert per_class["CHẨN_ĐOÁN"]["recall"] == 0.5
        assert per_class["CHẨN_ĐOÁN"]["precision"] == 1.0

    def test_per_class_f1_all_types(self):
        """Test all 5 entity types."""
        true_entities = [
            {"text": "ho", "start": 0, "end": 2, "type": "TRIỆU_CHỨNG"},
            {"text": "xét nghiệm", "start": 3, "end": 13, "type": "TÊN_XÉT_NGHIỆM"},
            {"text": "100 mg", "start": 14, "end": 19, "type": "KẾT_QUẢ_XÉT_NGHIỆM"},
            {"text": "viêm", "start": 20, "end": 25, "type": "CHẨN_ĐOÁN"},
            {"text": "thuốc", "start": 26, "end": 31, "type": "THUỐC"},
        ]
        pred_entities = true_entities.copy()  # Perfect match

        per_class = compute_per_class_f1(true_entities, pred_entities)

        expected_types = {
            "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
            "CHẨN_ĐOÁN", "THUỐC"
        }

        assert set(per_class.keys()) == expected_types
        for entity_type in expected_types:
            assert per_class[entity_type]["f1"] == 1.0

    def test_per_class_f1_zero_support(self):
        """Test per-class with no true entities."""
        true_entities = []
        pred_entities = [
            {"text": "ho", "start": 0, "end": 2, "type": "TRIỆU_CHỨNG"},
        ]

        per_class = compute_per_class_f1(true_entities, pred_entities)

        # Should still have entry for TRIỆU_CHỨNG
        assert "TRIỆU_CHỨNG" in per_class
        assert per_class["TRIỆU_CHỨNG"]["support"] == 0


class TestErrorAnalysis:
    """Test error analysis functions."""

    def test_analyze_errors_boundary(self):
        """Test boundary error detection."""
        true_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]
        pred_entities = [
            {"text": "viêm phổ", "start": 0, "end": 8, "type": "CHẨN_ĐOÁN"},
        ]

        boundary_errors, wrong_type_errors = analyze_errors(true_entities, pred_entities)

        assert boundary_errors == 1
        assert wrong_type_errors == 0

    def test_analyze_errors_wrong_type(self):
        """Test wrong type error detection."""
        true_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]
        pred_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "TRIỆU_CHỨNG"},
        ]

        boundary_errors, wrong_type_errors = analyze_errors(true_entities, pred_entities)

        assert boundary_errors == 0
        assert wrong_type_errors == 1

    def test_detailed_error_analysis(self):
        """Test detailed error analysis with examples."""
        true_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]
        pred_entities = [
            {"text": "viêm", "start": 0, "end": 4, "type": "CHẨN_ĐOÁN"},  # boundary error
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "TRIỆU_CHỨNG"},  # wrong type
        ]

        analysis = detailed_error_analysis(true_entities, pred_entities)

        assert analysis["num_correct"] == 0
        assert analysis["num_false_negatives"] == 1
        assert analysis["num_false_positives"] == 2
        assert len(analysis["false_negatives"]) == 1
        assert len(analysis["false_positives"]) == 2


class TestMetricsReport:
    """Test metrics report generation."""

    def test_print_metrics_report(self):
        """Test metrics report formatting."""
        metrics = {
            "precision": 0.75,
            "recall": 0.8,
            "entity_f1": 0.774,
            "num_correct": 3,
            "num_true": 4,
            "num_pred": 4,
            "per_class_f1": {
                "CHẨN_ĐOÁN": {"f1": 0.8, "support": 2},
                "THUỐC": {"f1": 0.75, "support": 2},
            },
            "boundary_errors": 1,
            "wrong_type_errors": 0,
        }

        report = print_metrics_report(metrics)

        assert "NER Evaluation Report" in report
        assert "Precision" in report
        assert "Recall" in report
        assert "F1" in report
        assert "Per-Class F1" in report
        assert "CHẨN_ĐOÁN" in report
        assert "THUỐC" in report


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_overlapping_entities(self):
        """Test with overlapping entities."""
        true_entities = [
            {"text": "viêm", "start": 0, "end": 4, "type": "CHẨN_ĐOÁN"},
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "TRIỆU_CHỨNG"},
        ]
        pred_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]

        metrics = compute_entity_metrics(true_entities, pred_entities)

        # Should handle without error
        assert "entity_f1" in metrics

    def test_duplicate_entities(self):
        """Test with duplicate entities in predictions."""
        true_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]
        pred_entities = [
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
            {"text": "viêm phổi", "start": 0, "end": 10, "type": "CHẨN_ĐOÁN"},
        ]

        metrics = compute_entity_metrics(true_entities, pred_entities)

        # Should deduplicate
        assert metrics["num_correct"] == 1
        assert metrics["num_pred"] == 1  # Deduplicated

    def test_unicode_text(self):
        """Test with Vietnamese unicode text."""
        true_entities = [
            {"text": "viêm phổi", "start": 0, "end": 9, "type": "CHẨN_ĐOÁN"},
            {"text": "Paracetamol", "start": 10, "end": 21, "type": "THUỐC"},
        ]
        pred_entities = [
            {"text": "viêm phổi", "start": 0, "end": 9, "type": "CHẨN_ĐOÁN"},
        ]

        metrics = compute_entity_metrics(true_entities, pred_entities)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
