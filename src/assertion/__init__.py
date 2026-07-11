"""
Assertion Detection Module

Detects assertion status (negated, historical, family) for medical entities.

Architecture:
    1. Rule cue detector (high precision)
    2. Sentence and clause segmentation
    3. Scope resolver
    4. Optional XLM-R multi-label classifier
    5. Ensemble rule + classifier

Entity Types:
    - TRIỆU_CHỨNG (Symptom)
    - CHẨN_ĐOÁN (Diagnosis)
    - THUỐC (Medication)
"""

from src.assertion.cues import (
    CueType,
    CueMatch,
    CueDefinition,
    NEGATION_CUE_DEFINITIONS,
    HISTORICAL_CUE_DEFINITIONS,
    FAMILY_CUE_DEFINITIONS,
    find_cue_matches,
    get_cue_type,
    CUE_PATTERNS,
    CONJUNCTION_PATTERNS,
    DELIMITER_PATTERN,
)

from src.assertion.scope import (
    ClauseBoundary,
    SentenceScope,
    EntityScope,
    ClauseSegmenter,
    resolve_entity_scope,
    apply_scope_rules,
    is_entity_in_scope,
    get_scope_representation,
    HISTORICAL_SECTION_PATTERNS,
)

from src.assertion.rules import (
    AssertionStatus,
    EntityAssertion,
    NegationRule,
    HistoricalRule,
    FamilyRule,
    AssertionRules,
    RuleBasedDetector,
    detect_assertions,
    detect_assertions_batch,
)

from src.assertion.ensemble import (
    EnsembleStrategy,
    AssertionConfig,
    ClassifierPrediction,
    AssertionResult,
    AssertionClassifier,
    AssertionEnsemble,
    create_ensemble,
)

__all__ = [
    # Cues
    "CueType",
    "CueMatch",
    "CueDefinition",
    "find_cue_matches",
    "get_cue_type",
    "NEGATION_CUE_DEFINITIONS",
    "HISTORICAL_CUE_DEFINITIONS",
    "FAMILY_CUE_DEFINITIONS",
    # Scope
    "ClauseBoundary",
    "SentenceScope",
    "EntityScope",
    "ClauseSegmenter",
    "resolve_entity_scope",
    "apply_scope_rules",
    "is_entity_in_scope",
    "get_scope_representation",
    "HISTORICAL_SECTION_PATTERNS",
    # Rules
    "AssertionStatus",
    "EntityAssertion",
    "NegationRule",
    "HistoricalRule",
    "FamilyRule",
    "AssertionRules",
    "RuleBasedDetector",
    "detect_assertions",
    "detect_assertions_batch",
    # Ensemble
    "EnsembleStrategy",
    "AssertionConfig",
    "ClassifierPrediction",
    "AssertionResult",
    "AssertionClassifier",
    "AssertionEnsemble",
    "create_ensemble",
]
