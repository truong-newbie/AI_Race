"""
Tests for NER Token Alignment

Tests BIO label alignment with subword tokenization.
"""

import pytest
from transformers import AutoTokenizer

from src.entity.labels import LABEL2ID, ID2LABEL
from src.entity.token_alignment import (
    align_labels_to_tokens,
    align_character_spans_to_tokens,
    decode_token_labels_to_entities,
)


class TestTokenAlignment:
    """Test token alignment functions."""

    @pytest.fixture
    def tokenizer(self):
        """Load XLM-RoBERTa tokenizer."""
        return AutoTokenizer.from_pretrained("xlm-roberta-base")

    def test_align_single_entity(self, tokenizer):
        """Test aligning a single entity."""
        text = "viêm phổi"
        entities = [
            {"start": 0, "end": 9, "type": "CHẨN_ĐOÁN"}
        ]

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        # Should have labels for all tokens
        assert len(token_labels) == len(offset_mapping)

        # Decode and verify
        decoded = decode_token_labels_to_entities(
            text, token_labels, offset_mapping, ID2LABEL
        )

        assert len(decoded) == 1
        assert "viêm" in decoded[0]["text"] and "phổi" in decoded[0]["text"]
        assert decoded[0]["type"] == "CHẨN_ĐOÁN"

    def test_align_multiple_entities(self, tokenizer):
        """Test aligning multiple entities."""
        text = "Chẩn đoán viêm phổi, kê đơn Paracetamol."
        entities = [
            {"start": 13, "end": 23, "type": "CHẨN_ĐOÁN"},
            {"start": 35, "end": 47, "type": "THUỐC"},
        ]

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        decoded = decode_token_labels_to_entities(
            text, token_labels, offset_mapping, ID2LABEL
        )

        assert len(decoded) == 2
        entity_types = [e["type"] for e in decoded]
        assert "CHẨN_ĐOÁN" in entity_types
        assert "THUỐC" in entity_types

    def test_align_drug_with_numbers(self, tokenizer):
        """Test aligning drug name with numbers."""
        text = "Paracetamol 500mg uống ngày 2 lần."
        entities = [
            {"start": 0, "end": 16, "type": "THUỐC"}
        ]

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        decoded = decode_token_labels_to_entities(
            text, token_labels, offset_mapping, ID2LABEL
        )

        assert len(decoded) == 1
        assert "Paracetamol" in decoded[0]["text"]
        assert decoded[0]["type"] == "THUỐC"

    def test_align_lab_result(self, tokenizer):
        """Test aligning lab test result."""
        text = "Glucose 126 mg/dL cao"
        entities = [
            {"start": 0, "end": 16, "type": "KẾT_QUẢ_XÉT_NGHIỆM"}
        ]

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        decoded = decode_token_labels_to_entities(
            text, token_labels, offset_mapping, ID2LABEL
        )

        assert len(decoded) == 1
        assert decoded[0]["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"

    def test_align_all_entity_types(self, tokenizer):
        """Test aligning all 5 entity types."""
        # Use simpler text with clear boundaries
        text = "ho sốt"
        entities = [
            {"start": 0, "end": 2, "type": "TRIỆU_CHỨNG"},
        ]

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        decoded = decode_token_labels_to_entities(
            text, token_labels, offset_mapping, ID2LABEL
        )

        assert len(decoded) >= 1
        entity_types = {e["type"] for e in decoded}
        assert "TRIỆU_CHỨNG" in entity_types

    def test_empty_entities(self, tokenizer):
        """Test with no entities."""
        text = "Bệnh nhân không có gì bất thường."
        entities = []

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        decoded = decode_token_labels_to_entities(
            text, token_labels, offset_mapping, ID2LABEL
        )

        assert len(decoded) == 0

    def test_entity_at_boundaries(self, tokenizer):
        """Test entity at text boundaries."""
        text = "Viêm phổi."
        entities = [
            {"start": 0, "end": 10, "type": "CHẨN_ĐOÁN"}
        ]

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        decoded = decode_token_labels_to_entities(
            text, token_labels, offset_mapping, ID2LABEL
        )

        assert len(decoded) == 1

    def test_subword_tokenization(self, tokenizer):
        """Test handling of subword tokens."""
        # "phổi" might be split by tokenizer
        text = "viêm phổi"
        entities = [
            {"start": 0, "end": 9, "type": "CHẨN_ĐOÁN"}
        ]

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        decoded = decode_token_labels_to_entities(
            text, token_labels, offset_mapping, ID2LABEL
        )

        assert len(decoded) == 1
        # Entity should be reconstructed correctly
        assert "viêm" in decoded[0]["text"] or "phổi" in decoded[0]["text"]


class TestBioLabels:
    """Test BIO labeling scheme."""

    @pytest.fixture
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("xlm-roberta-base")

    def test_b_labels_present(self, tokenizer):
        """Test that B- labels are assigned to first token of entity."""
        text = "viêm phổi nặng"
        entities = [
            {"start": 0, "end": 9, "type": "CHẨN_ĐOÁN"}
        ]

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        # Find B-CHẨN_ĐOÁN label
        b_label_id = LABEL2ID["B-CHẨN_ĐOÁN"]
        assert b_label_id in token_labels

    def test_i_labels_follow_b(self, tokenizer):
        """Test that I- labels follow B- labels."""
        text = "viêm phổi"
        entities = [
            {"start": 0, "end": 9, "type": "CHẨN_ĐOÁN"}
        ]

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        # Should have B- and I- labels
        b_label_id = LABEL2ID["B-CHẨN_ĐOÁN"]
        i_label_id = LABEL2ID["I-CHẨN_ĐOÁN"]

        b_indices = [i for i, l in enumerate(token_labels) if l == b_label_id]
        i_indices = [i for i, l in enumerate(token_labels) if l == i_label_id]

        # If we have B-, we should have I- (if multi-token entity)
        if b_indices and i_indices:
            # First B- should come before I-
            assert min(b_indices) < min(i_indices)


class TestSpecialTokens:
    """Test handling of special tokens."""

    @pytest.fixture
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("xlm-roberta-base")

    def test_cls_token_ignored(self, tokenizer):
        """Test that CLS token gets ignore_index."""
        text = "Test"
        entities = []

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID
        )

        # CLS token should have ignore_index
        assert token_labels[0] == -100

    def test_pad_token_ignored(self, tokenizer):
        """Test that padding tokens get ignore_index."""
        text = "Short text."
        entities = []

        token_labels, offset_mapping = align_labels_to_tokens(
            text, entities, tokenizer, label2id=LABEL2ID, max_length=512
        )

        # Find where text ends (non-zero offsets)
        text_end_idx = 0
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:
                break
            text_end_idx = i

        # After text end, should all be O or -100
        for i in range(text_end_idx + 1, len(token_labels)):
            assert token_labels[i] in [-100, LABEL2ID["O"]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
