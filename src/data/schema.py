"""
Data Schema Definitions

Định nghĩa schema cho dữ liệu training/validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import json


# =============================================================================
# Enums (reuse from main schema)
# =============================================================================

class EntityType(str, Enum):
    """5 loại entity trong bài toán."""
    TRIEU_CHUNG = "TRIỆU_CHỨNG"
    TEN_XET_NGHIEM = "TÊN_XÉT_NGHIỆM"
    KET_QUA_XET_NGHIEM = "KẾT_QUẢ_XÉT_NGHIỆM"
    CHAN_DOAN = "CHẨN_ĐOÁN"
    THUOC = "THUỐC"

    @classmethod
    def values(cls) -> List[str]:
        return [e.value for e in cls]


class AssertionType(str, Enum):
    """3 loại assertion."""
    NEGATED = "isNegated"
    FAMILY = "isFamily"
    HISTORICAL = "isHistorical"

    @classmethod
    def values(cls) -> List[str]:
        return [e.value for e in cls]


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Entity:
    """Entity trong sample data."""
    text: str
    start: int  # 0-based, inclusive
    end: int    # 0-based, exclusive
    type: str   # EntityType value
    assertions: List[str] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "type": self.type,
            "assertions": self.assertions,
            "candidates": self.candidates,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            text=data["text"],
            start=data["start"],
            end=data["end"],
            type=data["type"],
            assertions=data.get("assertions", []),
            candidates=data.get("candidates", []),
        )


@dataclass
class Sample:
    """Một sample trong dataset."""
    id: str
    text: str
    entities: List[Entity]
    source: str = "template"  # template, icd_linking, rxnorm_linking, manual
    review_status: str = "auto_validated"  # auto_validated, pending_review, reviewed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "source": self.source,
            "review_status": self.review_status,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        metadata = {k: v for k, v in data.items()
                   if k not in ["id", "text", "entities", "source", "review_status"]}
        return cls(
            id=data["id"],
            text=data["text"],
            entities=entities,
            source=data.get("source", "template"),
            review_status=data.get("review_status", "auto_validated"),
            metadata=metadata,
        )


@dataclass
class ICDLinkingSample:
    """Sample cho ICD-10 linking."""
    id: str
    query_text: str
    mention: str
    positive_code: str
    negative_codes: List[str]
    source: str = "synthetic_hard_negative"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query_text": self.query_text,
            "mention": self.mention,
            "positive_code": self.positive_code,
            "negative_codes": self.negative_codes,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ICDLinkingSample":
        return cls(
            id=data["id"],
            query_text=data["query_text"],
            mention=data["mention"],
            positive_code=data["positive_code"],
            negative_codes=data.get("negative_codes", []),
            source=data.get("source", "synthetic_hard_negative"),
        )


@dataclass
class RxNormLinkingSample:
    """Sample cho RxNorm linking."""
    id: str
    query_text: str
    mention: str
    positive_rxcui: str
    positive_name: str
    negative_rxcuis: List[str]
    negative_names: List[str]
    source: str = "synthetic_hard_negative"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query_text": self.query_text,
            "mention": self.mention,
            "positive_rxcui": self.positive_rxcui,
            "positive_name": self.positive_name,
            "negative_rxcuis": self.negative_rxcuis,
            "negative_names": self.negative_names,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RxNormLinkingSample":
        return cls(
            id=data["id"],
            query_text=data["query_text"],
            mention=data["mention"],
            positive_rxcui=data["positive_rxcui"],
            positive_name=data["positive_name"],
            negative_rxcuis=data.get("negative_rxcuis", []),
            negative_names=data.get("negative_names", []),
            source=data.get("source", "synthetic_hard_negative"),
        )


# =============================================================================
# Validation Set Template
# =============================================================================

@dataclass
class ValidationTemplate:
    """Template cho manual validation."""
    id: str
    text: str
    expected_entities: List[Dict[str, Any]]  # Without position, filled by annotator
    notes: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    category: str = ""  # symptom, diagnosis, drug, lab, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "expected_entities": self.expected_entities,
            "notes": self.notes,
            "difficulty": self.difficulty,
            "category": self.category,
        }


# =============================================================================
# Dataset Split
# =============================================================================

@dataclass
class DatasetSplit:
    """Kết quả split dataset."""
    train: List[Sample]
    dev: List[Sample]
    internal_test: List[Sample]

    def to_dicts(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "train": [s.to_dict() for s in self.train],
            "dev": [s.to_dict() for s in self.dev],
            "internal_test": [s.to_dict() for s in self.internal_test],
        }


# =============================================================================
# File I/O
# =============================================================================

def save_jsonl(path: str, samples: List[Dict], append: bool = False) -> None:
    """Save list of dicts to JSONL file."""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file to list of dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_samples(path: str, samples: List[Sample]) -> None:
    """Save samples to JSONL file."""
    save_jsonl(path, [s.to_dict() for s in samples])


def load_samples(path: str) -> List[Sample]:
    """Load samples from JSONL file."""
    return [Sample.from_dict(d) for d in load_jsonl(path)]


# =============================================================================
# Span Utilities
# =============================================================================

def verify_span(text: str, entity: Entity) -> bool:
    """Verify entity span matches text."""
    if entity.start < 0 or entity.end > len(text) or entity.start >= entity.end:
        return False
    return text[entity.start:entity.end] == entity.text


def verify_all_spans(sample: Sample) -> bool:
    """Verify all entities in sample have correct spans."""
    for entity in sample.entities:
        if not verify_span(sample.text, entity):
            return False
    return True


def find_span(text: str, entity_text: str, start_hint: int = 0) -> tuple[int, int]:
    """Find span position for entity text in text."""
    idx = text.find(entity_text, start_hint)
    if idx == -1:
        raise ValueError(f"Text '{entity_text}' not found in '{text}'")
    return (idx, idx + len(entity_text))


# =============================================================================
# Constants
# =============================================================================

VALID_ENTITY_TYPES = EntityType.values()
VALID_ASSERTION_TYPES = AssertionType.values()

ENTITY_TYPE_FOR_ASSERTIONS = {EntityType.TRIEU_CHUNG.value,
                             EntityType.CHAN_DOAN.value,
                             EntityType.THUOC.value}
ENTITY_TYPE_FOR_CANDIDATES = {EntityType.CHAN_DOAN.value,
                             EntityType.THUOC.value}
