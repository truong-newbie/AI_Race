"""
Assertion Detection Rules Engine

Module để detect assertions (isNegated, isFamily, isHistorical) từ văn bản y khoa.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Assertion Cue Patterns
# =============================================================================

# Negation cues in Vietnamese
NEGATION_CUES = [
    # Direct negation
    r"không\s+",
    r"chưa\s+",
    r"không\s+có\s+",
    r"ko\s+",
    r"k\s+",  # abbreviated

    # Medical negation terms
    r"loại\s+trừ\s+",
    r"loại\s+trừ\s+bệnh\s+",
    r"không\s+thấy\s+",
    r"chưa\s+ghi\s+nhận\s+",
    r"không\s+ghi\s+nhận\s+",
    r"âm\tính\s+",
    r"dương\s+tính\s+",  # actually positive, but context matters
    r"bình\tthường\s+",
    r"không\s+bình\tthường\s+",
    r"không\s+phải\s+",
    r"không\s+còn\s+",
    r"hết\s+",
    r"đã\s+hết\s+",
    r"tạm\s+thời\s+không\s+",

    # English (for mixed text)
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bnegative\b",
    r"\bruled\s+out\b",
    r"\babsent\b",
]

# Historical cues
HISTORICAL_CUES = [
    # Vietnamese
    r"tiền\s+sử\s+",
    r"có\s+tiền\s+sử\s+",
    r"tiền\s+sử\s+bệnh\s+",
    r"tiền\s+sử\s+dùng\s+",
    r"đã\s+từng\s+",
    r"đã\s+sử\s+dụng\s+",
    r"trước\s+đây\s+",
    r"trước\s+khi\s+",
    r"cũ\s+",
    r"cũ\s+có\s+",
    r"đã\s+có\s+",
    r"lâu\s+năm\s+",
    r"mắc\s+từ\s+trước\s+",
    r"tình\s+trạng\s+cũ\s+",

    # English
    r"\bhistory\s+of\b",
    r"\bhad\s+\w+ed\b",
    r"\bprevious\b",
    r"\bpast\b",
    r"\bformer\b",
    r"\bonce\s+had\b",
]

# Family history cues
FAMILY_CUES = [
    # Vietnamese
    r"bố\s+",
    r"mẹ\s+",
    r"cha\s+",
    r"ông\s+",
    r"bà\s+",
    r"anh\s+",
    r"chị\s+",
    r"em\s+",
    r"con\s+",
    r"người\s+nhà\s+",
    r"gia\s+đình\s+",
    r"gia\s+đình\s+có\s+",
    r"họ\s+hàng\s+",
    r"họ\s+hàng\s+bên\s+",
    r"ông\s+bà\s+",
    r"cha\s+mẹ\s+",
    r"anh\s+chị\s+em\s+",
    r"tiền\s+sử\s+gia\s+đình\s+",

    # English
    r"\bfather\b",
    r"\bmother\b",
    r"\bparent\b",
    r"\bsibling\b",
    r"\bfamily\b",
    r"\bfamily\s+history\b",
    r"\bhereditary\b",
]


# =============================================================================
# Scope Delimiters
# =============================================================================

# Clause/sentence delimiters
CLAUSE_DELIMITERS = [
    r"[,\.。;；!！?？]",
    r"\bnhưng\b",
    r"\btuy\s+nhiên\b",
    r"\bHOWEVER\b",
    r"\bAND\b",
    r"\bOR\b",
    r"\bcòn\s+",
    r"\btrong\s+khi\s+",
]

CLAUSE_PATTERN = re.compile('|'.join(CLAUSE_DELIMITERS), re.IGNORECASE)

# Compile negation patterns
NEGATION_PATTERN = re.compile('|'.join(NEGATION_CUES), re.IGNORECASE)
HISTORICAL_PATTERN = re.compile('|'.join(HISTORICAL_CUES), re.IGNORECASE)
FAMILY_PATTERN = re.compile('|'.join(FAMILY_CUES), re.IGNORECASE)


@dataclass
class AssertionMatch:
    """Kết quả detection của một assertion."""
    assertion_type: str  # isNegated, isFamily, isHistorical
    cue_text: str
    cue_start: int
    cue_end: int
    scope_start: int
    scope_end: int


@dataclass
class EntityAssertion:
    """Assertion result cho một entity."""
    entity_text: str
    entity_start: int
    entity_end: int
    is_negated: bool = False
    is_family: bool = False
    is_historical: bool = False

    def to_list(self) -> list[str]:
        """Convert assertions to list format."""
        result = []
        if self.is_negated:
            result.append("isNegated")
        if self.is_family:
            result.append("isFamily")
        if self.is_historical:
            result.append("isHistorical")
        return result


class AssertionDetector:
    """
    Rule-based assertion detector.

    Detects:
    - isNegated: Entity is negated (không, chưa, etc.)
    - isFamily: Entity related to family (bố, mẹ, etc.)
    - isHistorical: Entity related to history (tiền sử, đã từng, etc.)
    """

    def __init__(self):
        self.negation_pattern = NEGATION_PATTERN
        self.historical_pattern = HISTORICAL_PATTERN
        self.family_pattern = FAMILY_PATTERN
        self.clause_pattern = CLAUSE_PATTERN

    def detect(self, text: str, entity_start: int, entity_end: int) -> EntityAssertion:
        """
        Detect assertions for an entity at given position.

        Args:
            text: Full text
            entity_start: Start position of entity
            entity_end: End position of entity

        Returns:
            EntityAssertion với các flags được set
        """
        entity_text = text[entity_start:entity_end]
        result = EntityAssertion(
            entity_text=entity_text,
            entity_start=entity_start,
            entity_end=entity_end
        )

        # Determine clause scope
        scope_start, scope_end = self._get_clause_scope(
            text, entity_start, entity_end
        )

        # Get text within scope for pattern matching
        scope_text = text[scope_start:scope_end]

        # Check for negation
        result.is_negated = self._check_negation(text, entity_start, scope_start)

        # Check for historical
        result.is_historical = self._check_historical(text, entity_start, scope_start)

        # Check for family
        result.is_family = self._check_family(text, entity_start, scope_start)

        return result

    def detect_all(self, text: str, entities: list[dict]) -> list[EntityAssertion]:
        """
        Detect assertions for all entities.

        Args:
            text: Full text
            entities: List of entity dicts với 'text', 'position'

        Returns:
            List of EntityAssertion
        """
        results = []

        for entity in entities:
            pos = entity.get('position', [0, len(entity.get('text', ''))])
            result = self.detect(text, pos[0], pos[1])
            results.append(result)

        return results

    def _get_clause_scope(
        self, text: str, entity_start: int, entity_end: int
    ) -> tuple[int, int]:
        """
        Get the clause boundaries containing the entity.

        Returns:
            (scope_start, scope_end)
        """
        # Look backwards for clause delimiter
        scope_start = 0
        look_back = 150  # Max scope length
        search_start = max(0, entity_start - look_back)

        # Find last delimiter before entity
        last_delimiter = -1
        for match in self.clause_pattern.finditer(text[search_start:entity_start]):
            last_delimiter = search_start + match.start()

        if last_delimiter >= 0:
            scope_start = last_delimiter + 1

        # Look forwards for next delimiter
        scope_end = len(text)
        look_forward = 50
        search_end = min(len(text), entity_end + look_forward)

        next_delimiter = -1
        for match in self.clause_pattern.finditer(text[entity_end:search_end]):
            next_delimiter = entity_end + match.start()
            break

        if next_delimiter >= 0:
            scope_end = next_delimiter

        return (scope_start, scope_end)

    def _check_negation(self, text: str, entity_start: int, scope_start: int) -> bool:
        """
        Check if entity is negated.

        Checks if there's a negation cue between scope_start and entity_start.
        """
        # Get text between scope start and entity
        search_text = text[scope_start:entity_start]

        # Check for negation pattern
        match = self.negation_pattern.search(search_text)
        if match:
            return True

        return False

    def _check_historical(self, text: str, entity_start: int, scope_start: int) -> bool:
        """Check if entity is historical."""
        search_text = text[scope_start:entity_start]

        match = self.historical_pattern.search(search_text)
        return match is not None

    def _check_family(self, text: str, entity_start: int, scope_start: int) -> bool:
        """Check if entity is family-related."""
        search_text = text[scope_start:entity_start]

        match = self.family_pattern.search(search_text)
        return match is not None

    def get_cues(self, text: str) -> list[AssertionMatch]:
        """
        Get all assertion cues in text.

        Returns:
            List of AssertionMatch
        """
        cues = []

        # Find negation cues
        for match in self.negation_pattern.finditer(text):
            cues.append(AssertionMatch(
                assertion_type="isNegated",
                cue_text=match.group(),
                cue_start=match.start(),
                cue_end=match.end(),
                scope_start=max(0, match.start() - 150),
                scope_end=min(len(text), match.end() + 50)
            ))

        # Find historical cues
        for match in self.historical_pattern.finditer(text):
            cues.append(AssertionMatch(
                assertion_type="isHistorical",
                cue_text=match.group(),
                cue_start=match.start(),
                cue_end=match.end(),
                scope_start=max(0, match.start() - 150),
                scope_end=min(len(text), match.end() + 50)
            ))

        # Find family cues
        for match in self.family_pattern.finditer(text):
            cues.append(AssertionMatch(
                assertion_type="isFamily",
                cue_text=match.group(),
                cue_start=match.start(),
                cue_end=match.end(),
                scope_start=max(0, match.start() - 150),
                scope_end=min(len(text), match.end() + 50)
            ))

        return cues


# =============================================================================
# Utility Functions
# =============================================================================

def detect_assertions(
    text: str,
    entity_start: int,
    entity_end: int
) -> EntityAssertion:
    """
    Convenience function để detect assertions.

    Args:
        text: Full text
        entity_start: Start position
        entity_end: End position

    Returns:
        EntityAssertion
    """
    detector = AssertionDetector()
    return detector.detect(text, entity_start, entity_end)


def detect_assertions_batch(
    text: str,
    entities: list[dict]
) -> list[EntityAssertion]:
    """
    Convenience function để detect assertions for multiple entities.

    Args:
        text: Full text
        entities: List of entity dicts

    Returns:
        List of EntityAssertion
    """
    detector = AssertionDetector()
    return detector.detect_all(text, entities)


# =============================================================================
# Tests
# =============================================================================

def test_assertion_detector():
    """Test assertion detector."""
    detector = AssertionDetector()

    test_cases = [
        # (text, entity_pos, expected_assertions)
        ("Bệnh nhân không ho", (16, 19), ["isNegated"]),
        ("Bệnh nhân ho đờm xanh", (12, 19), []),
        ("Có tiền sử hen suyễn", (12, 23), ["isHistorical"]),
        ("Bố bệnh nhân bị đái tháo đường", (0, 3), ["isFamily"]),
        ("Bệnh nhân đã từng bị viêm phổi", (12, 23), ["isHistorical"]),
        ("Chưa ghi nhận bất thường", (20, 35), ["isNegated"]),
    ]

    print("=== Assertion Detection Tests ===\n")
    for text, (start, end), expected in test_cases:
        result = detector.detect(text, start, end)
        print(f"Text: '{text}'")
        print(f"Entity: '{text[start:end]}' at [{start}, {end}]")
        print(f"Expected: {expected}")
        print(f"Got: {result.to_list()}")
        print(f"Match: {result.to_list() == expected}")
        print()


if __name__ == "__main__":
    test_assertion_detector()
