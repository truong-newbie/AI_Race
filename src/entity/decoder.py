"""
NER Decoder Module

Decode BIO predictions back to entities.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch

from src.entity.labels import ID2LABEL, get_entity_type_from_label


class NERDecoder:
    """Decoder for converting BIO token predictions to entities."""

    def __init__(
        self,
        id2label: Dict[int, str] = None,
        confidence_threshold: float = 0.0,
    ):
        """Initialize decoder.

        Args:
            id2label: ID to label mapping
            confidence_threshold: Minimum confidence for entity
        """
        self.id2label = id2label or ID2LABEL
        self.confidence_threshold = confidence_threshold

    def decode(
        self,
        text: str,
        predictions: List[int],
        offset_mapping: List[Tuple[int, int]],
        logits: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Decode token predictions to entities.

        Args:
            text: Original text
            predictions: Token-level predictions (label IDs)
            offset_mapping: Token offset mapping from tokenizer
            logits: Optional token logits for confidence scores

        Returns:
            List of entity dicts
        """
        entities = []
        current_entity = None

        for token_idx, pred_id in enumerate(predictions):
            if token_idx >= len(offset_mapping):
                break

            label = self.id2label.get(pred_id, "O")
            char_start, char_end = offset_mapping[token_idx]

            # Skip special tokens
            if char_start == 0 and char_end == 0:
                continue

            # Calculate confidence
            confidence = 1.0
            if logits is not None and token_idx < len(logits):
                confidence = self._get_confidence(logits[token_idx], pred_id)

            # Check threshold
            if confidence < self.confidence_threshold:
                continue

            if label.startswith("B-"):
                # Save previous entity
                if current_entity is not None:
                    entities.append(current_entity)

                # Start new entity
                entity_type = label[2:]
                current_entity = {
                    "text": text[char_start:char_end],
                    "start": char_start,
                    "end": char_end,
                    "type": entity_type,
                    "confidence": confidence,
                    "tokens": [(token_idx, label)],
                }

            elif label.startswith("I-"):
                if current_entity is not None:
                    entity_type = label[2:]
                    # Check if same entity type
                    if current_entity["type"] == entity_type:
                        # Extend entity
                        current_entity["end"] = char_end
                        current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                        current_entity["tokens"].append((token_idx, label))
                        # Update confidence as running average
                        n = len(current_entity["tokens"])
                        current_entity["confidence"] = (
                            current_entity["confidence"] * (n - 1) + confidence
                        ) / n
                    else:
                        # Type mismatch - save old, start new
                        entities.append(current_entity)
                        current_entity = {
                            "text": text[char_start:char_end],
                            "start": char_start,
                            "end": char_end,
                            "type": entity_type,
                            "confidence": confidence,
                            "tokens": [(token_idx, label)],
                        }

            elif label == "O":
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget last entity
        if current_entity is not None:
            entities.append(current_entity)

        # Clean up tokens list
        for entity in entities:
            entity.pop("tokens", None)

        return entities

    def decode_batch(
        self,
        texts: List[str],
        predictions: torch.Tensor,
        offset_mappings: List[List[Tuple[int, int]]],
        logits: Optional[torch.Tensor] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Decode batch of predictions.

        Args:
            texts: List of original texts
            predictions: Batch predictions (batch_size, seq_len)
            offset_mappings: List of offset mappings
            logits: Optional batch logits

        Returns:
            List of entity lists
        """
        results = []

        for idx, text in enumerate(texts):
            preds = predictions[idx].tolist()
            offsets = offset_mappings[idx]

            logit = None
            if logits is not None:
                logit = logits[idx].numpy()

            entities = self.decode(text, preds, offsets, logit)
            results.append(entities)

        return results

    def _get_confidence(
        self,
        logits: np.ndarray,
        pred_id: int,
    ) -> float:
        """Get confidence score from logits.

        Args:
            logits: Token logits
            pred_id: Predicted label ID

        Returns:
            Confidence score (0-1)
        """
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return float(probs[pred_id])


def predictions_to_entities(
    text: str,
    predictions: List[int],
    offset_mapping: List[Tuple[int, int]],
    id2label: Dict[int, str] = None,
) -> List[Dict[str, Any]]:
    """Convert predictions to entities (convenience function).

    Args:
        text: Original text
        predictions: Token predictions
        offset_mapping: Token offset mapping
        id2label: ID to label mapping

    Returns:
        List of entities
    """
    decoder = NERDecoder(id2label=id2label)
    return decoder.decode(text, predictions, offset_mapping)


def extract_entities_from_model_output(
    text: str,
    logits: torch.Tensor,
    offset_mapping: List[Tuple[int, int]],
    confidence_threshold: float = 0.5,
    id2label: Dict[int, str] = None,
) -> List[Dict[str, Any]]:
    """Extract entities from model output.

    Args:
        text: Original text
        logits: Model logits (seq_len, num_labels)
        offset_mapping: Token offset mapping
        confidence_threshold: Minimum confidence
        id2label: ID to label mapping

    Returns:
        List of entities
    """
    id2label = id2label or ID2LABEL

    # Get predictions
    predictions = torch.argmax(logits, dim=-1).tolist()
    logits_np = logits.numpy()

    decoder = NERDecoder(
        id2label=id2label,
        confidence_threshold=confidence_threshold,
    )

    return decoder.decode(text, predictions, offset_mapping, logits_np)


# =============================================================================
# Test Functions
# =============================================================================

def test_basic_decode():
    """Test basic decoding."""
    decoder = NERDecoder()

    text = "Bệnh nhân bị viêm phổi."
    predictions = [
        0,  # O
        0,  # O
        0,  # O
        0,  # O
        11,  # B-CHẨN_ĐOÁN
        23,  # I-CHẨN_ĐOÁN
        0,  # O
    ]
    offset_mapping = [
        (0, 0),  # CLS
        (0, 5),  # Bệnh
        (6, 12),  # nhân
        (13, 16),  # bị
        (17, 22),  # viêm
        (22, 27),  # phổi
        (27, 28),  # .
    ]

    entities = decoder.decode(text, predictions, offset_mapping)

    assert len(entities) == 1
    assert entities[0]["type"] == "CHẨN_ĐOÁN"
    assert entities[0]["text"] == "viêm phổi"
    print("[OK] Basic decode test passed")


def test_multiple_entities_decode():
    """Test decoding multiple entities."""
    decoder = NERDecoder()

    text = "Chẩn đoán viêm phổi, kê đơn Paracetamol."
    predictions = [
        11,  # B-CHẨN_ĐOÁN
        23,  # I-CHẨN_ĐOÁN
        0,   # O
        15,  # B-THUỐC
        27,  # I-THUỐC
        27,  # I-THUỐC
        0,   # O
    ]
    offset_mapping = [
        (0, 6),   # Chẩn
        (6, 11),  # đoán
        (12, 22), # viêm phổi
        (24, 29), # kê đơn
        (30, 41), # Paracetamol
        (41, 42), # .
    ]

    entities = decoder.decode(text, predictions, offset_mapping)

    assert len(entities) == 2
    print(f"[OK] Multiple entities test: {len(entities)} entities found")


def test_consecutive_same_type():
    """Test consecutive I- labels of same type."""
    decoder = NERDecoder()

    text = "Glucose 126 mg/dL"
    predictions = [
        13,  # B-KẾT_QUẢ_XÉT_NGHIỆM
        25,  # I-KẾT_QUẢ_XÉT_NGHIỆM
        25,  # I-KẾT_QUẢ_XÉT_NGHIỆM
        25,  # I-KẾT_QUẢ_XÉT_NGHIỆM
        25,  # I-KẾT_QUẢ_XÉT_NGHIỆM
        25,  # I-KẾT_QUẢ_XÉT_NGHIỆM
    ]
    offset_mapping = [
        (0, 7),   # Glucose
        (8, 11),  # 126
        (12, 14), # mg
        (14, 17), # /dL
    ]

    entities = decoder.decode(text, predictions, offset_mapping)

    assert len(entities) == 1
    print(f"[OK] Consecutive same type test: entity = {entities[0]['text']}")


if __name__ == "__main__":
    print("Running decoder tests...")
    test_basic_decode()
    test_multiple_entities_decode()
    test_consecutive_same_type()
    print("All decoder tests passed!")
