"""
NER Label Definitions

BIO tagging scheme cho 5 entity types:
- O: Outside
- B-*: Beginning of entity
- I-*: Inside entity (continuation)
"""

from enum import Enum
from typing import List, Dict


# Entity types for NER
NER_ENTITY_TYPES = [
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
]

# BIO labels
LABEL_LIST = ["O"] + [
    f"B-{entity}"
    for entity in NER_ENTITY_TYPES
] + [
    f"I-{entity}"
    for entity in NER_ENTITY_TYPES
]

# Label to ID mapping
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Number of labels
NUM_LABELS = len(LABEL_LIST)


def get_entity_type_from_label(label: str) -> str | None:
    """Extract entity type from BIO label."""
    if label == "O":
        return None
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return None


def is_beginning_label(label: str) -> bool:
    """Check if label is a B- (beginning) label."""
    return label.startswith("B-")


def is_inside_label(label: str) -> bool:
    """Check if label is an I- (inside) label."""
    return label.startswith("I-")


def is_entity_label(label: str) -> bool:
    """Check if label represents an entity (B- or I-)."""
    return label != "O" and (label.startswith("B-") or label.startswith("I-"))


def create_label_mask(labels: List[int], ignore_index: int = -100) -> List[int]:
    """Create mask for labels that should be used in loss computation.

    Args:
        labels: List of label IDs
        ignore_index: Label ID to ignore in loss (e.g., special tokens)

    Returns:
        Mask where 1 means include in loss, 0 means ignore
    """
    return [0 if label == ignore_index else 1 for label in labels]
