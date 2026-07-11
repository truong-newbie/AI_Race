"""
Assertion Cues Module

Contains all assertion cue patterns for:
- isNegated
- isHistorical
- isFamily
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class CueType(Enum):
    """Types of assertion cues."""
    NEGATION = "isNegated"
    HISTORICAL = "isHistorical"
    FAMILY = "isFamily"


@dataclass
class CueDefinition:
    """Definition of an assertion cue."""
    pattern: str
    cue_type: CueType
    flags: int = re.IGNORECASE
    priority: int = 0  # Higher = checked first


# =============================================================================
# NEGATION CUES (isNegated)
# =============================================================================

NEGATION_CUE_DEFINITIONS = [
    # Core negation words
    CueDefinition(r"không\b", CueType.NEGATION, priority=10),
    CueDefinition(r"chưa\b", CueType.NEGATION, priority=10),
    CueDefinition(r"ko\b", CueType.NEGATION, priority=10),
    CueDefinition(r"k\b", CueType.NEGATION, priority=5),

    # "không có" pattern
    CueDefinition(r"không\s+có\b", CueType.NEGATION, priority=15),

    # "chưa ghi nhận" pattern
    CueDefinition(r"chưa\s+ghi\s+nhận\b", CueType.NEGATION, priority=20),
    CueDefinition(r"chưa\s+ghi\s+nhận\s+bệnh", CueType.NEGATION, priority=25),

    # "không ghi nhận" pattern
    CueDefinition(r"không\s+ghi\s+nhận\b", CueType.NEGATION, priority=20),
    CueDefinition(r"không\s+ghi\s+nhận\s+bệnh", CueType.NEGATION, priority=25),

    # "không thấy" pattern
    CueDefinition(r"không\s+thấy\b", CueType.NEGATION, priority=20),

    # "không xuất hiện" pattern
    CueDefinition(r"không\s+xuất\s+hiện\b", CueType.NEGATION, priority=20),

    # "loại trừ" pattern
    CueDefinition(r"loại\s+trừ\b", CueType.NEGATION, priority=20),
    CueDefinition(r"loại\s+trừ\s+bệnh", CueType.NEGATION, priority=25),

    # "âm tính" pattern
    CueDefinition(r"âm\s+tính\s+với\b", CueType.NEGATION, priority=25),
    CueDefinition(r"dương\s+tính\s+với\b", CueType.NEGATION, priority=15),  # Context matters

    # "không bình thường" / "bình thường"
    CueDefinition(r"không\s+bình\s+thường\b", CueType.NEGATION, priority=20),
    CueDefinition(r"bình\s+thường\b", CueType.NEGATION, priority=15),
    CueDefinition(r"bình\s+thường\s+về\b", CueType.NEGATION, priority=20),

    # "không phải"
    CueDefinition(r"không\s+phải\b", CueType.NEGATION, priority=20),

    # "không còn"
    CueDefinition(r"không\s+còn\b", CueType.NEGATION, priority=20),

    # "hết"
    CueDefinition(r"hết\s+", CueType.NEGATION, priority=15),
    CueDefinition(r"đã\s+hết\b", CueType.NEGATION, priority=20),

    # "tạm thời"
    CueDefinition(r"tạm\s+thời\s+không\b", CueType.NEGATION, priority=20),

    # "không đau" etc. - negation before symptom
    CueDefinition(r"không\s+(ho|sốt|đau|sốt|khó\s+thở|buồn|nôn|tiêu\s+chảy|mệt|chóng\s+mặt)", CueType.NEGATION, priority=25),

    # English terms (for mixed text)
    CueDefinition(r"\bno\b", CueType.NEGATION, priority=10),
    CueDefinition(r"\bnot\b", CueType.NEGATION, priority=10),
    CueDefinition(r"\bwithout\b", CueType.NEGATION, priority=10),
    CueDefinition(r"\bnegative\b", CueType.NEGATION, priority=15),
    CueDefinition(r"\bruled\s+out\b", CueType.NEGATION, priority=25),
    CueDefinition(r"\babsent\b", CueType.NEGATION, priority=15),
    CueDefinition(r"\bdenies\b", CueType.NEGATION, priority=20),
    CueDefinition(r"\bdenied\b", CueType.NEGATION, priority=20),
]


# =============================================================================
# HISTORICAL CUES (isHistorical)
# =============================================================================

HISTORICAL_CUE_DEFINITIONS = [
    # "tiền sử" pattern
    CueDefinition(r"tiền\s+sử\b", CueType.HISTORICAL, priority=20),
    CueDefinition(r"tiền\s+sử\s+bệnh\b", CueType.HISTORICAL, priority=25),
    CueDefinition(r"có\s+tiền\s+sử\b", CueType.HISTORICAL, priority=25),
    CueDefinition(r"có\s+tiền\s+sử\s+bệnh\b", CueType.HISTORICAL, priority=30),

    # "đã từng" pattern
    CueDefinition(r"đã\s+từng\b", CueType.HISTORICAL, priority=20),
    CueDefinition(r"từng\b", CueType.HISTORICAL, priority=15),

    # "trước đây" / "trước đó" pattern
    CueDefinition(r"trước\s+đây\b", CueType.HISTORICAL, priority=20),
    CueDefinition(r"trước\s+đó\b", CueType.HISTORICAL, priority=20),
    CueDefinition(r"trước\s+khi\b", CueType.HISTORICAL, priority=20),

    # "cách đây" pattern
    CueDefinition(r"cách\s+đây\b", CueType.HISTORICAL, priority=20),

    # "từng điều trị"
    CueDefinition(r"từng\s+điều\s+trị\b", CueType.HISTORICAL, priority=25),
    CueDefinition(r"đã\s+điều\s+trị\b", CueType.HISTORICAL, priority=25),

    # "đã sử dụng"
    CueDefinition(r"đã\s+sử\s+dụng\b", CueType.HISTORICAL, priority=25),
    CueDefinition(r"từng\s+sử\s+dụng\b", CueType.HISTORICAL, priority=25),

    # "bệnh cũ" / "cũ"
    CueDefinition(r"bệnh\s+cũ\b", CueType.HISTORICAL, priority=20),
    CueDefinition(r"tình\s+trạng\s+cũ\b", CueType.HISTORICAL, priority=25),
    CueDefinition(r"cũ\s+có\b", CueType.HISTORICAL, priority=20),
    CueDefinition(r"đã\s+có\b", CueType.HISTORICAL, priority=20),

    # "lâu năm" / "mắc từ trước"
    CueDefinition(r"lâu\s+năm\b", CueType.HISTORICAL, priority=15),
    CueDefinition(r"mắc\s+từ\s+trước\b", CueType.HISTORICAL, priority=20),
    CueDefinition(r"từ\s+lâu\b", CueType.HISTORICAL, priority=15),

    # English terms
    CueDefinition(r"\bhistory\s+of\b", CueType.HISTORICAL, priority=20),
    CueDefinition(r"\bhad\b", CueType.HISTORICAL, priority=15),
    CueDefinition(r"\bprevious\b", CueType.HISTORICAL, priority=15),
    CueDefinition(r"\bpast\b", CueType.HISTORICAL, priority=15),
    CueDefinition(r"\bformer\b", CueType.HISTORICAL, priority=15),
    CueDefinition(r"\bonce\s+had\b", CueType.HISTORICAL, priority=20),
    CueDefinition(r"\bwas\s+diagnosed\b", CueType.HISTORICAL, priority=20),
]


# =============================================================================
# FAMILY CUES (isFamily)
# =============================================================================

FAMILY_CUE_DEFINITIONS = [
    # Individual family relations
    CueDefinition(r"\bbố\b", CueType.FAMILY, priority=15),
    CueDefinition(r"\bmẹ\b", CueType.FAMILY, priority=15),
    CueDefinition(r"\bcha\b", CueType.FAMILY, priority=15),
    CueDefinition(r"\bông\b", CueType.FAMILY, priority=15),
    CueDefinition(r"\bbà\b", CueType.FAMILY, priority=15),

    # Siblings
    CueDefinition(r"\banh\b", CueType.FAMILY, priority=10),
    CueDefinition(r"\bchị\b", CueType.FAMILY, priority=10),
    CueDefinition(r"\bem\b", CueType.FAMILY, priority=10),
    CueDefinition(r"\bcon\b", CueType.FAMILY, priority=10),

    # Family group terms
    CueDefinition(r"người\s+nhà\b", CueType.FAMILY, priority=20),
    CueDefinition(r"gia\s+đình\b", CueType.FAMILY, priority=20),
    CueDefinition(r"gia\s+đình\s+có\b", CueType.FAMILY, priority=25),
    CueDefinition(r"họ\s+hàng\b", CueType.FAMILY, priority=20),
    CueDefinition(r"họ\s+hàng\s+bên\b", CueType.FAMILY, priority=25),
    CueDefinition(r"ông\s+bà\b", CueType.FAMILY, priority=20),
    CueDefinition(r"cha\s+mẹ\b", CueType.FAMILY, priority=25),
    CueDefinition(r"anh\s+chị\s+em\b", CueType.FAMILY, priority=25),

    # "tiền sử gia đình" - combined
    CueDefinition(r"tiền\s+sử\s+gia\s+đình\b", CueType.FAMILY, priority=30),
    CueDefinition(r"tiền\s+sử\s+gia\s+đình\s+bệnh\b", CueType.FAMILY, priority=35),

    # "bệnh nhân có người nhà"
    CueDefinition(r"bệnh\s+nhân\s+có\s+người\s+nhà\b", CueType.FAMILY, priority=25),

    # English terms
    CueDefinition(r"\bfather\b", CueType.FAMILY, priority=15),
    CueDefinition(r"\bmother\b", CueType.FAMILY, priority=15),
    CueDefinition(r"\bparent\b", CueType.FAMILY, priority=15),
    CueDefinition(r"\bsibling\b", CueType.FAMILY, priority=15),
    CueDefinition(r"\bfamily\b", CueType.FAMILY, priority=20),
    CueDefinition(r"\bfamily\s+history\b", CueType.FAMILY, priority=25),
    CueDefinition(r"\bhereditary\b", CueType.FAMILY, priority=25),
    CueDefinition(r"\bgenetic\b", CueType.FAMILY, priority=20),
]


# =============================================================================
# CONJUNCTION HANDLING
# =============================================================================

# Conjunctions that break scope / introduce new clauses
CLAUSE_CONJUNCTIONS = [
    (r"nhưng\b", "contrast"),           # but - contrast
    (r"tuy\s+nhiên\b", "contrast"),    # however
    (r"song\b", "contrast"),            # but/yet
    (r"hiện\s+tại\b", "current"),       # currently - marks current state
    (r"còn\b", "contrast"),             # while/but - marks contrast
    (r"tuy\b", "contrast"),             # though
    (r"mặc\s+dù\b", "contrast"),       # although
    (r"thế\s+mà\b", "contrast"),        # yet
    (r"đồng\s+thời\b", "concurrent"),  # simultaneously
]


# =============================================================================
# SCOPE DELIMITERS
# =============================================================================

# Sentence/phrase end markers
SCOPE_DELIMITERS = [
    r"[.。]",    # Period - end of sentence
    r"[;；]",    # Semicolon - clause break
    r"[!！]",    # Exclamation - strong end
    r"[?？]",    # Question - clause break
]


# =============================================================================
# COMPILED PATTERNS
# =============================================================================

def compile_patterns():
    """Compile all cue patterns for efficient matching."""
    patterns = []

    for cue_def in NEGATION_CUE_DEFINITIONS:
        try:
            pattern = re.compile(cue_def.pattern, cue_def.flags)
            patterns.append((pattern, cue_def.cue_type, cue_def.priority, cue_def.pattern))
        except re.error:
            continue

    for cue_def in HISTORICAL_CUE_DEFINITIONS:
        try:
            pattern = re.compile(cue_def.pattern, cue_def.flags)
            patterns.append((pattern, cue_def.cue_type, cue_def.priority, cue_def.pattern))
        except re.error:
            continue

    for cue_def in FAMILY_CUE_DEFINITIONS:
        try:
            pattern = re.compile(cue_def.pattern, cue_def.flags)
            patterns.append((pattern, cue_def.cue_type, cue_def.priority, cue_def.pattern))
        except re.error:
            continue

    # Sort by priority (higher first)
    patterns.sort(key=lambda x: -x[2])

    return patterns


def compile_conjunctions():
    """Compile conjunction patterns."""
    result = []
    for pattern, conj_type in CLAUSE_CONJUNCTIONS:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            result.append((compiled, conj_type))
        except re.error:
            continue
    return result


def compile_delimiters():
    """Compile scope delimiter patterns."""
    pattern = "|".join(SCOPE_DELIMITERS)
    return re.compile(pattern, re.IGNORECASE)


# Pre-compiled patterns
CUE_PATTERNS = compile_patterns()
CONJUNCTION_PATTERNS = compile_conjunctions()
DELIMITER_PATTERN = compile_delimiters()


# =============================================================================
# CUE MATCH DATA CLASS
# =============================================================================

@dataclass
class CueMatch:
    """A matched assertion cue."""
    cue_type: CueType
    text: str
    start: int
    end: int
    priority: int
    original_pattern: str


# =============================================================================
# CUE FINDING FUNCTIONS
# =============================================================================

def find_cue_matches(text: str) -> List[CueMatch]:
    """
    Find all assertion cue matches in text.

    Args:
        text: Input text

    Returns:
        List of CueMatch sorted by position
    """
    matches = []

    for pattern, cue_type, priority, orig_pattern in CUE_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(CueMatch(
                cue_type=cue_type,
                text=match.group(),
                start=match.start(),
                end=match.end(),
                priority=priority,
                original_pattern=orig_pattern
            ))

    # Sort by position
    matches.sort(key=lambda x: x.start)
    return matches


def get_cue_type_name(cue_type: CueType) -> str:
    """Get string name of cue type."""
    return cue_type.value


def get_cue_type(cue_type: CueType) -> str:
    """Alias for get_cue_type_name."""
    return get_cue_type_name(cue_type)
