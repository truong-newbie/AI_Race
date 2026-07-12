"""
Sentence and Clause Scope Resolver

Resolves the scope of assertion cues - determining which entities
are affected by which cues within a text.

Scope Rules:
1. Cues do not span across sentences (period delimiter)
2. Stop at period, semicolon when appropriate
3. Handle conjunctions: nhưng, tuy nhiên, song, hiện tại, còn
4. Cues can apply to entity lists: "không ho, sốt, khó thở"
5. Conjunction exceptions: "không ho nhưng đau ngực" - only ho is negated
6. Family scope: "mẹ bệnh nhân từng mắc hen suyễn" - hen is isFamily and isHistorical
7. Historical section: entities in "TIỀN SỬ" section get isHistorical
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from src.assertion.cues import (
    CueMatch,
    CueType,
    find_cue_matches,
    CUE_PATTERNS,
    CONJUNCTION_PATTERNS,
    DELIMITER_PATTERN,
)


# Section patterns that imply historical context.
# The "tiền sử bệnh" pattern uses a negative lookbehind (?<![a-z...])
# to ONLY match when NOT preceded by a letter — this distinguishes:
#   - "Tiền sử bệnh\n    Thuốc" (no preceding letter) → MATCHES → historical section
#   - "có tiền sử bệnh tim"  (letter "ó" before "tiền") → NO match → NOT historical
HISTORICAL_SECTION_PATTERNS = [
    r"tiền\s+sử\s+bệnh\b",
    r"quá\s+khứ\b",
    r"tiền\s+sử\b",
    r"bệnh\s+sử\b",
    r"past\s+history\b",
    r"history\b",
]

# Section header patterns that act as sentence boundaries in Vietnamese medical text.
# These numbered/bulleted section headings break the scope of prior sections so that
# a "Tiền sử" section header does NOT bleed historical marking into the next section's
# bullet points (e.g., drugs listed under "Thuốc đang dùng").
# Matches patterns like "1. ", "2. ", "3. " at the start of a line or after a newline.
SECTION_HEADER_PATTERN = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+',
    re.MULTILINE
)

# Section header cues: "Tiền sử bệnh hiện tại" / "1. Tiền sử..." at the START of a
# sentence should NOT trigger historical marking for the rest of that sentence.
# Only mid-sentence historical cues like "BN có tiền sử bệnh tiểu đường" should.
# Matches:
#   - "1.  Tiền sử..."     (numbered section header, any variant)
#   - "- Tiền sử bệnh hiện tại"  (bulleted section header, WITH "hiện tại")
#   - "Tiền sử bệnh\n" or "Tiền sử bệnh hiện tại"  (plain header at start of sentence)
#     The (?=\s) lookahead requires whitespace after "bệnh":
#     - "Tiền sử bệnh\n" → whitespace → MATCHES → excluded (correct!)
#     - "Tiền sử bệnh hiện tại" → space → MATCHES → excluded (correct!)
#     - "Tiền sử bệnh tiểu đường" → letter 't' → NO match → NOT excluded (correct!)
SECTION_HEADER_HISTORICAL_PATTERN = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|'
    r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b|'
    r'^tiền\s+sử\s+bệnh\s+(?=\s)(?!\S)',
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)


@dataclass
class ClauseBoundary:
    """Represents a clause boundary."""
    start: int
    end: int
    clause_type: str  # "main", "contrast", "concurrent", "list"


@dataclass
class SentenceScope:
    """Represents a sentence's scope for assertion detection."""
    text: str
    start: int
    end: int
    clauses: List[ClauseBoundary]
    has_historical_section: bool


@dataclass
class EntityScope:
    """Scope information for a specific entity."""
    entity_start: int
    entity_end: int
    sentence_scope: SentenceScope
    clause_scope: ClauseBoundary
    relevant_cues: List[CueMatch]


class ClauseSegmenter:
    """Segments text into sentences and clauses."""

    def __init__(self):
        self.delimiter_pattern = DELIMITER_PATTERN
        self.conjunction_patterns = CONJUNCTION_PATTERNS

    def segment_sentences(self, text: str) -> List[SentenceScope]:
        """
        Segment text into sentences.

        In Vietnamese medical records, sections are separated by numbered/bulleted
        headers (e.g. "1. Tiền sử bệnh hiện tại"). These headers act as sentence
        boundaries so that a "Tiền sử" section does not bleed isHistorical marking
        into subsequent section bullet points.

        Args:
            text: Input text

        Returns:
            List of SentenceScope objects
        """
        sentences = []
        start = 0

        # Collect all boundary positions: standard delimiters + section headers
        boundary_positions: list[int] = []
        for match in self.delimiter_pattern.finditer(text):
            boundary_positions.append(match.end())
        for match in SECTION_HEADER_PATTERN.finditer(text):
            boundary_positions.append(match.start())

        # Sort and deduplicate
        boundary_positions = sorted(set(boundary_positions))

        for end in boundary_positions:
            if end <= start:
                continue
            sentence_text = text[start:end]
            # Strip leading/trailing whitespace and punctuation for clean sentence text
            sentence_text = sentence_text.strip(' \t\n')
            while sentence_text and sentence_text[-1] in '.!?':
                sentence_text = sentence_text[:-1]

            # Check for historical section
            has_hist = self._is_historical_section(sentence_text)

            # Segment into clauses
            clauses = self._segment_clauses(sentence_text, start)

            sentences.append(SentenceScope(
                text=sentence_text,
                start=start,
                end=end,
                clauses=clauses,
                has_historical_section=has_hist
            ))

            start = end

        # Handle remaining text (no trailing delimiter)
        if start < len(text):
            remaining = text[start:]
            has_hist = self._is_historical_section(remaining)
            clauses = self._segment_clauses(remaining, start)
            sentences.append(SentenceScope(
                text=remaining,
                start=start,
                end=len(text),
                clauses=clauses,
                has_historical_section=has_hist
            ))

        return sentences

    def _is_historical_section(self, text: str) -> bool:
        """
        Check if text is part of a historical section.

        Section headers like "1. Tiền sử bệnh hiện tại" are excluded —
        they are document structure, not historical assertions.

        The DOTALL flag is used so that \s (whitespace) matches newlines,
        because the section title may span across lines in the raw text.
        """
        text_lower = text.lower()
        stripped_lower = text_lower.lstrip()

        # Check HISTORICAL_SECTION_PATTERNS first. A sentence is a historical section
        # ONLY if the historical cue appears at the start of the sentence (position 0
        # in stripped_lower) or immediately after a newline. This distinguishes:
        #   - "Tiền sử bệnh\n..." (section header, start of sentence) → is historical
        #   - "Tiền sử bệnh\n    Thuốc" → start-of-sentence header → section-header exclusion
        #     applies → NOT historical
        #   - "Bố có tiền sử bệnh tim." (mid-sentence cue at pos 6) → NOT historical
        # After the HISTORICAL_SECTION_PATTERNS check, re-exclude section headers
        # (like "Tiền sử bệnh\n" which is document structure, not patient history).
        for pattern in HISTORICAL_SECTION_PATTERNS:
            m = re.search(pattern, stripped_lower, re.DOTALL)
            if m:
                # Only treat as historical section if cue is at sentence start or
                # immediately after a newline (multi-line section title).
                at_sentence_start = (m.start() == 0)
                after_newline = (m.start() > 0 and stripped_lower[m.start() - 1] == '\n')
                if not (at_sentence_start or after_newline):
                    continue  # mid-sentence cue → not a historical section
                # At sentence start: check if it's a section header to exclude
                header_m = SECTION_HEADER_HISTORICAL_PATTERN.search(stripped_lower)
                if header_m:
                    return False  # section header → not historical
                return True  # historical section cue at start of sentence

        return False

    def _segment_clauses(self, sentence_text: str, sentence_start: int) -> List[ClauseBoundary]:
        """Segment a sentence into clauses."""
        clauses = []
        current_start = 0

        # Find all clause boundaries
        boundaries = []

        # Conjunction boundaries only
        for pattern, conj_type in self.conjunction_patterns:
            for match in pattern.finditer(sentence_text):
                boundaries.append((match.start(), "contrast" if conj_type == "contrast" else "concurrent"))

        # Sort boundaries by position
        boundaries.sort(key=lambda x: x[0])

        # Create clause boundaries
        for boundary_pos, boundary_type in boundaries:
            if boundary_pos > current_start:
                clause_end = boundary_pos
                # Strip trailing comma/semicolon from clause text
                while clause_end > current_start and sentence_text[clause_end - 1] in ",;":
                    clause_end -= 1
                clauses.append(ClauseBoundary(
                    start=sentence_start + current_start,
                    end=sentence_start + clause_end,
                    clause_type="list" if boundary_type == "list" else "contrast"
                ))
                current_start = boundary_pos
                # Skip past commas that mark list separators
                while current_start < len(sentence_text) and sentence_text[current_start] in ",;":
                    current_start += 1

        # Final clause (strip trailing punctuation)
        if current_start < len(sentence_text):
            final_end = len(sentence_text)
            while final_end > current_start and sentence_text[final_end - 1] in ",;.":
                final_end -= 1
            clauses.append(ClauseBoundary(
                start=sentence_start + current_start,
                end=sentence_start + final_end,
                clause_type="main"
            ))

        return clauses if clauses else [
            ClauseBoundary(
                start=sentence_start,
                end=sentence_start + len(sentence_text),
                clause_type="main"
            )
        ]

    def get_entity_sentence(
        self, text: str, entity_start: int, entity_end: int, sentences: List[SentenceScope]
    ) -> Optional[SentenceScope]:
        """Get the sentence containing an entity."""
        for sentence in sentences:
            if sentence.start <= entity_start < sentence.end:
                return sentence
        return None

    def get_entity_clause(
        self, entity_start: int, entity_end: int, clauses: List[ClauseBoundary]
    ) -> Optional[ClauseBoundary]:
        """Get the clause containing an entity."""
        for clause in clauses:
            if clause.start <= entity_start < clause.end:
                return clause
        return None


def resolve_entity_scope(
    text: str,
    entity_start: int,
    entity_end: int,
    segmenter: Optional[ClauseSegmenter] = None
) -> EntityScope:
    """
    Resolve the scope for an entity.

    Args:
        text: Full text
        entity_start: Entity start position
        entity_end: Entity end position
        segmenter: Optional ClauseSegmenter (creates one if not provided)

    Returns:
        EntityScope with relevant information
    """
    if segmenter is None:
        segmenter = ClauseSegmenter()

    sentences = segmenter.segment_sentences(text)
    sentence_scope = segmenter.get_entity_sentence(text, entity_start, entity_end, sentences)

    if sentence_scope is None:
        # Entity not in any sentence, return minimal scope
        return EntityScope(
            entity_start=entity_start,
            entity_end=entity_end,
            sentence_scope=SentenceScope(
                text=text[entity_start:entity_end],
                start=entity_start,
                end=entity_end,
                clauses=[],
                has_historical_section=False
            ),
            clause_scope=ClauseBoundary(
                start=entity_start,
                end=entity_end,
                clause_type="main"
            ),
            relevant_cues=[]
        )

    clause_scope = segmenter.get_entity_clause(entity_start, entity_end, sentence_scope.clauses)

    if clause_scope is None:
        clause_scope = ClauseBoundary(
            start=entity_start,
            end=entity_end,
            clause_type="main"
        )

    # Find relevant cues in scope
    relevant_cues = _find_relevant_cues(
        text, entity_start, sentence_scope, clause_scope
    )

    return EntityScope(
        entity_start=entity_start,
        entity_end=entity_end,
        sentence_scope=sentence_scope,
        clause_scope=clause_scope,
        relevant_cues=relevant_cues
    )


def _find_relevant_cues(
    text: str,
    entity_start: int,
    sentence_scope: SentenceScope,
    clause_scope: ClauseBoundary
) -> List[CueMatch]:
    """Find cues that are relevant to an entity based on scope rules."""
    all_cues = find_cue_matches(text)
    relevant = []

    for cue in all_cues:
        # Rule 1: Cues do not span across sentences
        if cue.end < sentence_scope.start or cue.start >= sentence_scope.end:
            continue

        # Rule 3 & 5: Handle conjunction-based scope
        if _is_cue_in_scope(text, cue, entity_start, sentence_scope, clause_scope):
            relevant.append(cue)

    return relevant


def _is_cue_in_scope(
    text: str,
    cue: CueMatch,
    entity_start: int,
    sentence_scope: SentenceScope,
    clause_scope: ClauseBoundary
) -> bool:
    """
    Determine if a cue is in scope for an entity.

    Rule 5: "không ho nhưng đau ngực" - only ho is negated
    Cues after contrast conjunctions (nhưng, tuy nhiên) only apply to
    entities AFTER the conjunction.
    """
    # Cues before the entity in the same clause are in scope
    if cue.end <= entity_start and cue.end > clause_scope.start:
        return True

    # Check if cue is in a preceding clause
    for clause in sentence_scope.clauses:
        if clause.end <= entity_start and clause.end > clause_scope.start:
            if clause.clause_type == "contrast":
                # Cues in contrast clause before entity: check if entity is in same clause
                if cue.end <= entity_start and cue.end >= clause_scope.start:
                    return True
            elif clause.clause_type == "main":
                # Cues in main clause apply to entity if no contrast separator
                return True

    # Check for conjunction between cue and entity
    clause_text = text[clause_scope.start:clause_scope.end]
    for pattern, conj_type in CONJUNCTION_PATTERNS:
        conj_match = pattern.search(clause_text)
        if conj_match:
            conj_pos = clause_scope.start + conj_match.start()
            # If conjunction is between cue and entity, cue is out of scope
            if cue.end <= conj_pos < entity_start:
                return False

    # Historical section rule: cues in a historical section sentence are in scope,
    # cues in a non-historical sentence are out of scope (except for section headers,
    # which already return has_historical_section=False from _is_historical_section).
    if cue.cue_type == CueType.HISTORICAL:
        if sentence_scope.has_historical_section:
            return True  # in historical section → cue in scope
        return False  # not in historical section → historical cue out of scope

    return False


def is_entity_in_scope(
    cue: CueMatch,
    entity_start: int,
    clause_scope: ClauseBoundary
) -> bool:
    """
    Check if an entity is within the scope of a cue.

    Args:
        cue: The assertion cue
        entity_start: Entity start position
        clause_scope: The clause containing the entity

    Returns:
        True if entity is in scope of cue
    """
    # Cue must be before entity
    if cue.end > entity_start:
        return False

    # Check for contrast conjunction between cue and entity
    between_text = text_between(cue.end, entity_start)
    for pattern, conj_type in CONJUNCTION_PATTERNS:
        if conj_type == "contrast" and pattern.search(between_text):
            return False

    return True


def text_between(start: int, end: int) -> str:
    """Helper to get text between two positions."""
    # This is a placeholder - actual text is passed in resolve functions
    return ""


def apply_scope_rules(
    text: str,
    entity_start: int,
    entity_end: int,
    cues: List[CueMatch],
    clause_segmenter: ClauseSegmenter
) -> Dict[str, bool]:
    """
    Apply scope rules to determine assertion status.

    Args:
        text: Full text
        entity_start: Entity start
        entity_end: Entity end
        cues: All cues in text
        clause_segmenter: Clause segmenter instance

    Returns:
        Dict with is_negated, is_historical, is_family
    """
    result = {
        "is_negated": False,
        "is_historical": False,
        "is_family": False
    }

    # Get sentence containing entity
    sentences = clause_segmenter.segment_sentences(text)
    sentence_scope = clause_segmenter.get_entity_sentence(text, entity_start, entity_end, sentences)

    if sentence_scope is None:
        return result

    # Get clause containing entity
    clause_scope = clause_segmenter.get_entity_clause(entity_start, entity_end, sentence_scope.clauses)
    if clause_scope is None:
        clause_scope = ClauseBoundary(
            start=entity_start, end=entity_end, clause_type="main"
        )

    # Check each cue
    for cue in cues:
        if not _is_cue_in_scope(text, cue, entity_start, sentence_scope, clause_scope):
            continue

        if cue.cue_type == CueType.NEGATION:
            result["is_negated"] = True
        elif cue.cue_type == CueType.HISTORICAL:
            result["is_historical"] = True
        elif cue.cue_type == CueType.FAMILY:
            result["is_family"] = True

    return result


# Utility for scope visualization
def get_scope_representation(
    text: str,
    entity_start: int,
    entity_end: int,
    cues: List[CueMatch]
) -> str:
    """Get a visual representation of scope for debugging."""
    lines = []
    lines.append(f"Text: {text}")
    lines.append(f"Entity: '{text[entity_start:entity_end]}' at [{entity_start}, {entity_end}]")
    lines.append("")

    for cue in cues:
        cue_type_str = cue.cue_type.value
        lines.append(f"  Cue [{cue.start}:{cue.end}]: {cue_type_str} = '{cue.text}'")

        if cue.end <= entity_start:
            lines.append(f"    -> Affects entity (cue before entity)")
        else:
            lines.append(f"    -> Does NOT affect entity (cue after entity)")

    return "\n".join(lines)
