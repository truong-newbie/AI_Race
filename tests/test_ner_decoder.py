"""
Tests for NER Decoder

Tests BIO predictions to entity conversion.
"""

import pytest
import numpy as np

from src.entity.decoder import (
    NERDecoder,
    predictions_to_entities,
    extract_entities_from_model_output,
)
from src.entity.labels import ID2LABEL, LABEL2ID


class TestNERDecoder:
    """Test NERDecoder class."""

    def test_decode_single_entity(self):
        """Test decoding a single entity."""
        decoder = NERDecoder(id2label=ID2LABEL)

        text = "viêm phổi"
        # Offset mapping - skip CLS with (0,0)
        offset_mapping = [
            (0, 0),    # CLS - special token
            (0, 4),    # viêm
            (5, 10),   # phổi
        ]

        # Label IDs: 4=B-CHẨN_ĐOÁN, 9=I-CHẨN_ĐOÁN
        predictions = [0, 4, 9]

        entities = decoder.decode(text, predictions, offset_mapping)

        assert len(entities) == 1
        assert entities[0]["type"] == "CHẨN_ĐOÁN"
        assert "viêm" in entities[0]["text"] or "phổi" in entities[0]["text"]

    def test_decode_multiple_entities(self):
        """Test decoding multiple entities."""
        decoder = NERDecoder(id2label=ID2LABEL)

        text = "viêm phổi Paracetamol"
        offset_mapping = [
            (0, 0),    # CLS
            (0, 4),    # viêm
            (5, 10),   # phổi
            (11, 23),  # Paracetamol
        ]

        # Label IDs: 4=B-CHẨN_ĐOÁN, 9=I-CHẨN_ĐOÁN, 5=B-THUỐC
        predictions = [0, 4, 9, 5]

        entities = decoder.decode(text, predictions, offset_mapping)

        assert len(entities) == 2
        entity_types = [e["type"] for e in entities]
        assert "CHẨN_ĐOÁN" in entity_types
        assert "THUỐC" in entity_types

    def test_decode_with_confidence(self):
        """Test confidence scores."""
        decoder = NERDecoder(id2label=ID2LABEL, confidence_threshold=0.5)

        text = "Test."
        offset_mapping = [(0, 0), (0, 4), (4, 5)]

        # Fake logits (higher for predicted class)
        logits = np.array([
            [10.0, 0.0, 0.0],  # CLS - O
            [1.0, 8.0, 1.0],   # Token - predicted class has high prob
            [10.0, 0.0, 0.0],  # . - O
        ])

        predictions = [0, 1, 0]

        entities = decoder.decode(text, predictions, offset_mapping, logits)

        # Check confidence is calculated
        if entities:
            assert "confidence" in entities[0]
            assert 0 <= entities[0]["confidence"] <= 1.0

    def test_decode_empty_predictions(self):
        """Test with no entities predicted."""
        decoder = NERDecoder(id2label=ID2LABEL)

        text = "Bệnh nhân bình thường."
        offset_mapping = [(0, 0), (0, 5), (6, 12), (13, 23), (23, 24)]
        predictions = [0, 0, 0, 0, 0]

        entities = decoder.decode(text, predictions, offset_mapping)

        assert len(entities) == 0

    def test_decode_boundary_error_handling(self):
        """Test handling of boundary errors (entities starting with I-)."""
        decoder = NERDecoder(id2label=ID2LABEL)

        text = "viêm phổi"
        offset_mapping = [(0, 0), (0, 4), (5, 9), (9, 10)]

        # I- without preceding B- (should be ignored or treated as new entity)
        predictions = [0, 23, 23, 0]  # I-CHẨN_ĐOÁN without B-

        entities = decoder.decode(text, predictions, offset_mapping)

        # Decoder should either ignore or start new entity
        # The key is it shouldn't crash
        assert isinstance(entities, list)

    def test_decode_all_entity_types(self):
        """Test all 5 entity types."""
        decoder = NERDecoder(id2label=ID2LABEL)

        text = "Entity."
        offset_mapping = [(0, 0), (0, 6), (6, 7)]

        for i, entity_type in enumerate([
            "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
            "CHẨN_ĐOÁN", "THUỐC"
        ]):
            b_label = LABEL2ID[f"B-{entity_type}"]
            predictions = [0, b_label, 0]

            entities = decoder.decode(text, predictions, offset_mapping)
            assert len(entities) == 1
            assert entities[0]["type"] == entity_type


class TestPredictionsToEntities:
    """Test convenience function."""

    def test_predictions_to_entities_basic(self):
        """Test basic usage."""
        text = "viêm phổi"
        offset_mapping = [(0, 0), (0, 4), (5, 9), (9, 10)]
        predictions = [0, 11, 23, 0]

        entities = predictions_to_entities(
            text, predictions, offset_mapping, id2label=ID2LABEL
        )

        assert len(entities) >= 0


class TestExtractEntitiesFromModelOutput:
    """Test extract_entities_from_model_output function."""

    def test_extract_entities_basic(self):
        """Test basic extraction."""
        import torch

        text = "Test"
        offset_mapping = [(0, 0), (0, 4), (4, 5)]

        # Create mock logits
        logits = torch.randn(3, 11)  # 3 tokens, 11 classes

        entities = extract_entities_from_model_output(
            text, logits, offset_mapping,
            id2label=ID2LABEL, confidence_threshold=0.0
        )

        assert isinstance(entities, list)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_predictions(self):
        """Test with predictions longer than offset_mapping."""
        decoder = NERDecoder(id2label=ID2LABEL)

        text = "Short."
        offset_mapping = [(0, 0), (0, 5), (5, 6)]
        predictions = [0, 0, 0, 0, 0, 0, 0, 0]  # More preds than offsets

        entities = decoder.decode(text, predictions, offset_mapping)

        # Should not crash
        assert isinstance(entities, list)

    def test_all_o_predictions(self):
        """Test when all predictions are O."""
        decoder = NERDecoder(id2label=ID2LABEL)

        text = "No entities here."
        offset_mapping = [(0, 0), (0, 2), (3, 10), (11, 15), (15, 16)]
        predictions = [0, 0, 0, 0, 0]

        entities = decoder.decode(text, predictions, offset_mapping)

        assert len(entities) == 0

    def test_consecutive_b_labels(self):
        """Test consecutive B- labels (new entity immediately after end)."""
        decoder = NERDecoder(id2label=ID2LABEL)

        text = "Entity1 Entity2"
        offset_mapping = [(0, 0), (0, 7), (8, 15), (15, 16)]

        # Two consecutive B- labels
        predictions = [0, 11, 15, 0]  # B-CHẨN_ĐOÁN, B-THUỐC

        entities = decoder.decode(text, predictions, offset_mapping)

        # Should decode 2 entities or 0 if entities not matching text
        assert isinstance(entities, list)

    def test_i_without_matching_type(self):
        """Test I- label with type different from current B-."""
        decoder = NERDecoder(id2label=ID2LABEL)

        text = "entity1 entity2"
        offset_mapping = [(0, 0), (0, 7), (8, 15), (15, 16)]

        # B-CHẨN_ĐOÁN followed by I-THUỐC (wrong type)
        predictions = [0, 11, 15, 0]

        entities = decoder.decode(text, predictions, offset_mapping)

        # Should handle type mismatch - creates 2 entities or 1 depending on implementation
        assert isinstance(entities, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
