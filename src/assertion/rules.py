"""
Rule-Based Assertion Detector

Detects assertions using rule-based pattern matching.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from src.assertion.cues import (
    CueType,
    CueMatch,
    CueDefinition,
    find_cue_matches,
    NEGATION_CUE_DEFINITIONS,
    HISTORICAL_CUE_DEFINITIONS,
    FAMILY_CUE_DEFINITIONS,
    CUE_PATTERNS,
    CONJUNCTION_PATTERNS,
)
from src.assertion.scope import (
    ClauseSegmenter,
    resolve_entity_scope,
    apply_scope_rules,
)

# Section header patterns that are NOT historical cues.
# "Tiền sử bệnh hiện tại" in Vietnamese medical records is a section header
# (e.g. "1. Tiền sử bệnh hiện tại"), not a historical assertion about the patient.
SECTION_HEADER_CUE_PATTERN = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b',
    re.IGNORECASE | re.MULTILINE
)


@dataclass
class AssertionStatus:
    """Assertion status for an entity."""
    is_negated: bool = False
    is_historical: bool = False
    is_family: bool = False
    confidence: float = 1.0
    source: str = "rule"
    cues_used: List[str] = field(default_factory=list)


@dataclass
class EntityAssertion:
    """Complete assertion result for an entity."""
    entity_text: str
    entity_start: int
    entity_end: int
    entity_type: Optional[str] = None
    status: AssertionStatus = field(default_factory=AssertionStatus)
    scope_info: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "entity_text": self.entity_text,
            "entity_start": self.entity_start,
            "entity_end": self.entity_end,
            "entity_type": self.entity_type,
            "is_negated": self.status.is_negated,
            "is_historical": self.status.is_historical,
            "is_family": self.status.is_family,
            "confidence": self.status.confidence,
            "source": self.status.source,
        }

    def to_list(self) -> List[str]:
        """Convert assertions to list format."""
        result = []
        if self.status.is_negated:
            result.append("isNegated")
        if self.status.is_family:
            result.append("isFamily")
        if self.status.is_historical:
            result.append("isHistorical")
        return result


class NegationRule:
    """Negation detection rule."""

    def __init__(self):
        self.cue_type = CueType.NEGATION

    def apply(
        self,
        text: str,
        entity_start: int,
        entity_end: int,
        cues: List[CueMatch],
        clause_segmenter: ClauseSegmenter
    ) -> Tuple[bool, float, List[str]]:
        """
        Apply negation rule.

        Returns:
            (is_negated, confidence, cues_used)
        """
        # Apply scope rules
        result = apply_scope_rules(text, entity_start, entity_end, cues, clause_segmenter)

        if not result["is_negated"]:
            return False, 1.0, []

        # Calculate confidence based on cue priority
        cues_used = []
        max_priority = 0
        for cue in cues:
            if cue.cue_type == CueType.NEGATION:
                cues_used.append(cue.text)
                max_priority = max(max_priority, cue.priority)

        # Confidence based on cue specificity
        confidence = min(0.7 + (max_priority / 100), 0.99) if max_priority > 0 else 0.8

        return True, confidence, cues_used


class HistoricalRule:
    """Historical detection rule."""

    def __init__(self):
        self.cue_type = CueType.HISTORICAL

    def apply(
        self,
        text: str,
        entity_start: int,
        entity_end: int,
        cues: List[CueMatch],
        clause_segmenter: ClauseSegmenter
    ) -> Tuple[bool, float, List[str]]:
        """
        Apply historical rule.

        Returns:
            (is_historical, confidence, cues_used)
        """
        # Filter out section header cues — "Tiền sử bệnh hiện tại" is a section title,
        # not a historical assertion about the patient.
        historical_cues = [
            c for c in cues
            if c.cue_type == CueType.HISTORICAL
            and not SECTION_HEADER_CUE_PATTERN.search(text[max(0, c.start - 20):c.end + 20])
        ]

        # Apply scope rules with filtered cues
        result = apply_scope_rules(text, entity_start, entity_end, historical_cues, clause_segmenter)

        if not result["is_historical"]:
            return False, 1.0, []

        # Rule: family history context — "Bố có tiền sử bệnh tim" → isFamily only, not isHistorical.
        # When a historical cue ("có tiền sử", "tiền sử bệnh", etc.) is immediately preceded
        # by a family cue with a small gap, the historical marking is suppressed.
        # This distinguishes "Bố có tiền sử bệnh tim" (family member's history)
        # from "bệnh nhân có tiền sử bệnh tim" (patient's own history → isHistorical).
        # Note: search over ALL cues (not historical_cues) because historical_cues only
        # contains HISTORICAL-type cues; family cues were filtered out.
        for cue in historical_cues:
            if cue.cue_type == CueType.HISTORICAL:
                cue_text = cue.text.strip()
                # Only apply this exception for "tiền sử" family history patterns
                if cue_text.startswith("tiền sử") or cue_text.startswith("có tiền sử"):
                    for fam_cue in cues:  # search ALL cues
                        if fam_cue.cue_type == CueType.FAMILY:
                            gap = cue.start - fam_cue.end
                            if 0 < gap <= 10:  # Family cue directly before tiền sử
                                return False, 1.0, []

        # Calculate confidence
        cues_used = []
        max_priority = 0
        for cue in historical_cues:
            if cue.cue_type == CueType.HISTORICAL:
                cues_used.append(cue.text)
                max_priority = max(max_priority, cue.priority)

        confidence = min(0.7 + (max_priority / 100), 0.99) if max_priority > 0 else 0.8

        return True, confidence, cues_used


class FamilyRule:
    """Family history detection rule."""

    def __init__(self):
        self.cue_type = CueType.FAMILY

    def apply(
        self,
        text: str,
        entity_start: int,
        entity_end: int,
        cues: List[CueMatch],
        clause_segmenter: ClauseSegmenter
    ) -> Tuple[bool, float, List[str]]:
        """
        Apply family rule.

        Returns:
            (is_family, confidence, cues_used)
        """
        # Apply scope rules
        result = apply_scope_rules(text, entity_start, entity_end, cues, clause_segmenter)

        if not result["is_family"]:
            return False, 1.0, []

        # Calculate confidence
        cues_used = []
        max_priority = 0
        for cue in cues:
            if cue.cue_type == CueType.FAMILY:
                cues_used.append(cue.text)
                max_priority = max(max_priority, cue.priority)

        confidence = min(0.7 + (max_priority / 100), 0.99) if max_priority > 0 else 0.8

        return True, confidence, cues_used


class AssertionRules:
    """Container for all assertion rules."""

    def __init__(self):
        self.negation_rule = NegationRule()
        self.historical_rule = HistoricalRule()
        self.family_rule = FamilyRule()
        self.clause_segmenter = ClauseSegmenter()

    def detect(
        self,
        text: str,
        entity_start: int,
        entity_end: int,
        entity_type: Optional[str] = None
    ) -> EntityAssertion:
        """Detect all assertions for an entity."""
        entity_text = text[entity_start:entity_end]

        # Find all cues in text
        all_cues = find_cue_matches(text)

        # Apply each rule
        is_negated, neg_conf, neg_cues = self.negation_rule.apply(
            text, entity_start, entity_end, all_cues, self.clause_segmenter
        )

        is_historical, hist_conf, hist_cues = self.historical_rule.apply(
            text, entity_start, entity_end, all_cues, self.clause_segmenter
        )

        is_family, fam_conf, fam_cues = self.family_rule.apply(
            text, entity_start, entity_end, all_cues, self.clause_segmenter
        )

        # Combine confidences
        all_cues_used = neg_cues + hist_cues + fam_cues
        if all_cues_used:
            confidence = min(neg_conf, hist_conf, fam_conf) if any([is_negated, is_historical, is_family]) else 1.0
        else:
            confidence = 1.0

        status = AssertionStatus(
            is_negated=is_negated,
            is_historical=is_historical,
            is_family=is_family,
            confidence=confidence,
            source="rule",
            cues_used=all_cues_used
        )

        return EntityAssertion(
            entity_text=entity_text,
            entity_start=entity_start,
            entity_end=entity_end,
            entity_type=entity_type,
            status=status
        )


class RuleBasedDetector:
    """Rule-based assertion detector - main entry point."""

    def __init__(self):
        self.rules = AssertionRules()

    def detect(
        self,
        text: str,
        entity_start: int,
        entity_end: int,
        entity_type: Optional[str] = None
    ) -> EntityAssertion:
        """Detect assertions for a single entity."""
        return self.rules.detect(text, entity_start, entity_end, entity_type)

    def detect_all(
        self,
        text: str,
        entities: List[dict]
    ) -> List[EntityAssertion]:
        """
        Detect assertions for multiple entities.

        Args:
            text: Full text
            entities: List of entity dicts with 'start', 'end', 'type' (optional)

        Returns:
            List of EntityAssertion
        """
        results = []
        for entity in entities:
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            entity_type = entity.get("type")

            result = self.detect(text, start, end, entity_type)
            results.append(result)

        return results

    def get_cues(self, text: str) -> List[CueMatch]:
        """Get all assertion cues in text."""
        return find_cue_matches(text)


# Keep backward compatibility with existing code
AssertionDetector = RuleBasedDetector
AssertionMatch = CueMatch


def detect_assertions(
    text: str,
    entity_start: int,
    entity_end: int,
    entity_type: Optional[str] = None
) -> EntityAssertion:
    """Convenience function for assertion detection."""
    detector = RuleBasedDetector()
    return detector.detect(text, entity_start, entity_end, entity_type)


def detect_assertions_batch(
    text: str,
    entities: List[dict]
) -> List[EntityAssertion]:
    """Convenience function for batch assertion detection."""
    detector = RuleBasedDetector()
    return detector.detect_all(text, entities)
