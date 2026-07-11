"""
Confidence Configuration and Scoring

Manages confidence thresholds and scoring policies for ensemble extraction.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class EntitySource(str, Enum):
    """Entity source types."""
    REGEX = "regex"
    DICTIONARY = "dictionary"
    NER_MODEL = "ner_model"
    MERGED = "merged"


# Default confidence scores by source
DEFAULT_SOURCE_CONFIDENCE = {
    EntitySource.REGEX: 0.85,
    EntitySource.DICTIONARY: 0.90,
    EntitySource.NER_MODEL: 0.75,  # Will be overridden by model confidence
    EntitySource.MERGED: 0.95,
}


@dataclass
class ConfidenceConfig:
    """Configuration for confidence scoring."""

    # Base thresholds
    ner_threshold: float = 0.5
    dictionary_confidence: float = 0.85
    regex_confidence: float = 0.80
    merged_confidence: float = 0.90

    # Bonuses
    agreement_bonus: float = 0.10  # Bonus when multiple sources agree
    context_bonus: float = 0.05   # Bonus from context matching

    # Thresholds
    overlap_threshold: float = 0.5  # IoU threshold for overlap detection
    type_conflict_margin: float = 0.1  # Margin for type conflict resolution

    # Section-based context weights
    section_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "chẩn đoán": {
            "CHẨN_ĐOÁN": 1.2,
            "TRIỆU_CHỨNG": 0.8,
        },
        "triệu chứng": {
            "TRIỆU_CHỨNG": 1.2,
            "CHẨN_ĐOÁN": 0.8,
        },
        "điều trị": {
            "THUỐC": 1.2,
        },
        "xét nghiệm": {
            "TÊN_XÉT_NGHIỆM": 1.2,
            "KẾT_QUẢ_XÉT_NGHIỆM": 1.1,
        },
        "kết quả": {
            "KẾT_QUẢ_XÉT_NGHIỆM": 1.2,
        },
    })

    # Drug-specific settings
    drug_prefer_dosage: bool = True
    drug_prefer_form: bool = True

    # Never merge these entity types
    never_merge_pairs: tuple = (
        ("TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"),
        ("KẾT_QUẢ_XÉT_NGHIỆM", "TÊN_XÉT_NGHIỆM"),
    )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ConfidenceConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ner_threshold": self.ner_threshold,
            "dictionary_confidence": self.dictionary_confidence,
            "regex_confidence": self.regex_confidence,
            "merged_confidence": self.merged_confidence,
            "agreement_bonus": self.agreement_bonus,
            "context_bonus": self.context_bonus,
            "overlap_threshold": self.overlap_threshold,
            "type_conflict_margin": self.type_conflict_margin,
        }


def compute_entity_confidence(
    entity: Dict[str, Any],
    config: ConfidenceConfig,
    source_scores: Optional[Dict[str, float]] = None,
) -> float:
    """Compute final confidence for an entity.

    Args:
        entity: Entity dict
        config: Confidence config
        source_scores: Optional per-source scores

    Returns:
        Computed confidence score
    """
    base_confidence = entity.get("confidence", 0.5)
    source = entity.get("source", EntitySource.NER_MODEL)

    # Apply source-specific base confidence
    source_base = {
        EntitySource.REGEX: config.regex_confidence,
        EntitySource.DICTIONARY: config.dictionary_confidence,
        EntitySource.NER_MODEL: config.ner_threshold,
        EntitySource.MERGED: config.merged_confidence,
    }.get(source, 0.5)

    # Start with base confidence
    confidence = base_confidence

    # Add agreement bonus if multiple sources agree
    if source_scores and len(source_scores) > 1:
        confidence += config.agreement_bonus

    # Apply context bonus if available
    if "context_bonus" in entity:
        confidence += entity["context_bonus"] * config.context_bonus

    # Cap at 1.0
    return min(confidence, 1.0)


def get_source_base_confidence(source: EntitySource, config: ConfidenceConfig) -> float:
    """Get base confidence for a source type."""
    return {
        EntitySource.REGEX: config.regex_confidence,
        EntitySource.DICTIONARY: config.dictionary_confidence,
        EntitySource.NER_MODEL: config.ner_threshold,
        EntitySource.MERGED: config.merged_confidence,
    }.get(source, 0.5)


def get_section_type_weight(
    entity_type: str,
    section: Optional[str],
    config: ConfidenceConfig,
) -> float:
    """Get type weight based on section context.

    Args:
        entity_type: Entity type
        section: Current section/section name
        config: Confidence config

    Returns:
        Weight multiplier (1.0 = neutral)
    """
    if not section:
        return 1.0

    section_lower = section.lower()

    # Check each section's type weights
    for section_key, type_weights in config.section_weights.items():
        if section_key in section_lower:
            return type_weights.get(entity_type, 1.0)

    return 1.0


def should_never_merge(type1: str, type2: str, config: ConfidenceConfig) -> bool:
    """Check if two entity types should never be merged."""
    return (type1, type2) in config.never_merge_pairs
