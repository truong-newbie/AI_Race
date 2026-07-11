"""
Token Alignment Module

Chuyển đổi character spans → token labels với offset_mapping.
Handles subword tokenization correctly.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from transformers import PreTrainedTokenizer


def align_labels_to_tokens(
    text: str,
    entities: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    label2id: Dict[str, int] = None,
    ignore_index: int = -100,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Align character-level entity labels to token-level labels.

    Args:
        text: Input text
        entities: List of entities with 'start', 'end', 'type'
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        label2id: Label to ID mapping
        ignore_index: ID for special tokens to ignore

    Returns:
        Tuple of (token_labels, offset_mapping)
    """
    if label2id is None:
        from src.entity.labels import LABEL2ID
        label2id = LABEL2ID

    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors=None,
    )

    # Get token labels
    token_labels = align_character_spans_to_tokens(
        text=text,
        entities=entities,
        offset_mapping=encoding["offset_mapping"],
        label2id=label2id,
        ignore_index=ignore_index,
    )

    return token_labels, encoding["offset_mapping"]


def align_character_spans_to_tokens(
    text: str,
    entities: List[Dict[str, Any]],
    offset_mapping: List[Tuple[int, int]],
    label2id: Dict[str, int],
    ignore_index: int = -100,
) -> List[int]:
    """Convert character spans to token labels.

    Handles subword tokenization:
    - First subword of entity gets B- label
    - Remaining subwords get I- label
    - Special tokens and subwords outside entities get O label

    Args:
        text: Original text
        entities: List of entities with 'start', 'end', 'type'
        offset_mapping: Token offset mapping from tokenizer
        label2id: Label to ID mapping
        ignore_index: ID for special tokens

    Returns:
        List of token labels
    """
    # Initialize all labels as 'O'
    token_labels = [label2id["O"]] * len(offset_mapping)

    # Sort entities by start position for proper handling
    sorted_entities = sorted(entities, key=lambda e: e["start"])

    # Create entity span coverage tracking
    entity_spans = []
    for entity in sorted_entities:
        start, end = entity["start"], entity["end"]
        entity_type = entity["type"]
        entity_spans.append((start, end, entity_type))

    # Assign labels to tokens based on character coverage
    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
        # Skip special tokens (cls, sep, pad, etc.)
        if char_start == 0 and char_end == 0:
            token_labels[token_idx] = ignore_index
            continue

        # Find if token is inside any entity
        assigned_label = None
        for ent_start, ent_end, ent_type in entity_spans:
            # Check if token overlaps with entity
            if char_start < ent_end and char_end > ent_start:
                # Token overlaps with entity
                # Check if this is the first token of the entity
                if char_start <= ent_start:
                    assigned_label = f"B-{ent_type}"
                else:
                    assigned_label = f"I-{ent_type}"
                break

        if assigned_label is not None and assigned_label in label2id:
            token_labels[token_idx] = label2id[assigned_label]
        else:
            token_labels[token_idx] = label2id["O"]

    return token_labels


def decode_token_labels_to_entities(
    text: str,
    token_labels: List[int],
    offset_mapping: List[Tuple[int, int]],
    id2label: Dict[int, str],
    confidence_threshold: float = 0.0,
    logits: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Convert token labels back to character span entities.

    Args:
        text: Original text for extracting entity text
        token_labels: Token-level predictions
        offset_mapping: Token offset mapping
        id2label: ID to label mapping
        confidence_threshold: Minimum confidence to include entity
        logits: Optional logits for confidence scores

    Returns:
        List of entity dicts with text, start, end, type, confidence
    """
    entities = []
    current_entity = None

    for token_idx, label_id in enumerate(token_labels):
        if token_idx >= len(offset_mapping):
            break

        label = id2label.get(label_id, "O")
        char_start, char_end = offset_mapping[token_idx]

        # Skip special tokens
        if char_start == 0 and char_end == 0:
            continue

        # Calculate confidence if logits provided
        confidence = 1.0
        if logits is not None and token_idx < len(logits):
            probs = softmax(logits[token_idx])
            confidence = float(probs[label_id])

        if label.startswith("B-"):
            # Save previous entity if exists
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
            }

        elif label.startswith("I-") and current_entity is not None:
            # Extend current entity
            # Check if entity type matches
            entity_type = label[2:]
            if current_entity["type"] == entity_type:
                current_entity["end"] = char_end
                current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                # Update confidence as average
                current_entity["confidence"] = (
                    current_entity["confidence"] + confidence
                ) / 2
            else:
                # Type mismatch - treat as new entity
                entities.append(current_entity)
                current_entity = {
                    "text": text[char_start:char_end],
                    "start": char_start,
                    "end": char_end,
                    "type": entity_type,
                    "confidence": confidence,
                }

        elif label == "O" and current_entity is not None:
            # End current entity
            entities.append(current_entity)
            current_entity = None

    # Don't forget last entity
    if current_entity is not None:
        entities.append(current_entity)

    # Filter by confidence threshold
    if confidence_threshold > 0:
        entities = [e for e in entities if e["confidence"] >= confidence_threshold]

    return entities


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def test_vietnamese_diacritics():
    """Test token alignment with Vietnamese diacritics."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    text = "Bệnh nhân bị viêm phổi nặng."
    entities = [
        {"start": 15, "end": 26, "type": "CHẨN_ĐOÁN"}
    ]

    from src.entity.labels import LABEL2ID

    token_labels, offset_mapping = align_labels_to_tokens(
        text, entities, tokenizer, label2id=LABEL2ID
    )

    # Verify entity span
    decoded = decode_token_labels_to_entities(
        text, token_labels, offset_mapping, {v: k for k, v in LABEL2ID.items()}
    )

    assert len(decoded) == 1
    assert decoded[0]["text"] == "viêm phổi"
    print("[OK] Vietnamese diacritics test passed")


def test_drug_with_numbers():
    """Test token alignment with drug names containing numbers."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    text = "Paracetamol 500mg uống ngày 2 lần."
    entities = [
        {"start": 0, "end": 16, "type": "THUỐC"}
    ]

    from src.entity.labels import LABEL2ID

    token_labels, offset_mapping = align_labels_to_tokens(
        text, entities, tokenizer, label2id=LABEL2ID
    )

    decoded = decode_token_labels_to_entities(
        text, token_labels, offset_mapping, {v: k for k, v in LABEL2ID.items()}
    )

    assert len(decoded) == 1
    assert decoded[0]["text"] == "Paracetamol 500mg"
    print("[OK] Drug with numbers test passed")


def test_multiple_entities():
    """Test token alignment with multiple entities in one sentence."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    text = "Chẩn đoán viêm phổi, kê đơn Paracetamol."
    entities = [
        {"start": 13, "end": 23, "type": "CHẨN_ĐOÁN"},
        {"start": 35, "end": 47, "type": "THUỐC"},
    ]

    from src.entity.labels import LABEL2ID

    token_labels, offset_mapping = align_labels_to_tokens(
        text, entities, tokenizer, label2id=LABEL2ID
    )

    decoded = decode_token_labels_to_entities(
        text, token_labels, offset_mapping, {v: k for k, v in LABEL2ID.items()}
    )

    assert len(decoded) == 2
    print("[OK] Multiple entities test passed")


def test_mg_ml_units():
    """Test token alignment with MG/ML units."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    text = "Glucose 126 mg/dL, Cholesterol 245 mg/dL."
    entities = [
        {"start": 0, "end": 31, "type": "KẾT_QUẢ_XÉT_NGHIỆM"},
        {"start": 33, "end": 55, "type": "KẾT_QUẢ_XÉT_NGHIỆM"},
    ]

    from src.entity.labels import LABEL2ID

    token_labels, offset_mapping = align_labels_to_tokens(
        text, entities, tokenizer, label2id=LABEL2ID
    )

    decoded = decode_token_labels_to_entities(
        text, token_labels, offset_mapping, {v: k for k, v in LABEL2ID.items()}
    )

    assert len(decoded) == 2
    print("[OK] MG/ML units test passed")


if __name__ == "__main__":
    print("Running token alignment tests...")
    test_vietnamese_diacritics()
    test_drug_with_numbers()
    test_multiple_entities()
    test_mg_ml_units()
    print("All tests passed!")
