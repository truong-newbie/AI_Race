"""
Ontology constraint validator for ICD-10 and RxNorm reranking.

Enforces rules derived from medical ontology structure:
- No child code when context lacks detail
- No candidate with exclude-term contradiction
- RxNorm: strength mismatch must not be masked by dense similarity
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConstraintResult:
    """Result of a constraint check."""
    passed: bool
    rule: str
    reason: str
    penalty: float = 0.0
    boost: float = 0.0


# --- ICD-10 Constraints ---


def check_icd_child_context(
    code: str,
    parent_code: Optional[str],
    mention_has_detail: bool,
    entry_name: str,
    entry_description: Optional[str],
) -> ConstraintResult:
    """
    Penalize child codes when context lacks detail.

    Rule: If mention is brief (e.g. "đau bụng") and the candidate is a
    child code (has parent_code != itself), prefer the parent code.
    Parent codes are more general and appropriate for vague mentions.
    """
    is_child = parent_code is not None and parent_code != code

    if not is_child:
        return ConstraintResult(passed=True, rule="icd_child_context", reason="not a child code")

    # Count detail signals in mention (approximate)
    detail_signals = ["mg", "mg/", "ml", "units", "iu", "lần", "ngày", "buổi",
                      "liều", "uống", "tiêm", "thuốc", "viên", "gói"]

    # Context has detail if:
    # 1. Mention mentions detail signals (drug-specific)
    # 2. OR entry description is long (semantically rich)
    has_detail = mention_has_detail or (entry_description and len(entry_description) > 80)

    if is_child and not has_detail:
        return ConstraintResult(
            passed=False,
            rule="icd_child_context",
            reason=f"child code '{code}' without detail in mention — prefer parent",
            penalty=0.25,
        )

    return ConstraintResult(passed=True, rule="icd_child_context", reason="child code with sufficient context")


def check_icd_exclude_terms(
    mention_normalized: str,
    exclude_terms: list[str],
) -> ConstraintResult:
    """
    Reject candidates where mention contains an exclude term.

    Rule: If mention text matches any exclude term, the candidate is
    invalid for this mention regardless of retrieval score.
    """
    if not exclude_terms:
        return ConstraintResult(passed=True, rule="icd_exclude_terms", reason="no exclude terms")

    mention_lower = mention_normalized.lower()

    for term in exclude_terms:
        term_lower = term.lower().strip()
        if term_lower and term_lower in mention_lower:
            return ConstraintResult(
                passed=False,
                rule="icd_exclude_terms",
                reason=f"mention contains exclude term '{term}'",
                penalty=1.0,
            )

    return ConstraintResult(passed=True, rule="icd_exclude_terms", reason="no exclude term match")


def check_icd_include_terms(
    mention_normalized: str,
    include_terms: list[str],
    alias_texts: list[str],
) -> ConstraintResult:
    """
    Boost candidates where mention contains an include term.

    Rule: Include terms are semantically defining features. If mention
    matches an include term, it strongly supports this candidate.
    """
    if not include_terms:
        return ConstraintResult(passed=True, rule="icd_include_terms", reason="no include terms")

    mention_lower = mention_normalized.lower()

    matched = []
    for term in include_terms:
        term_lower = term.lower().strip()
        if term_lower and term_lower in mention_lower:
            matched.append(term)

    if matched:
        # Boost proportional to fraction of include terms matched
        coverage = len(matched) / len(include_terms)
        boost = 0.15 * coverage + 0.05 * min(len(matched), 3)
        return ConstraintResult(
            passed=True,
            rule="icd_include_terms",
            reason=f"matched include terms: {matched}",
            boost=min(boost, 0.25),
        )

    return ConstraintResult(passed=True, rule="icd_include_terms", reason="no include term match")


# --- RxNorm Constraints ---


def check_rxnorm_strength_mismatch(
    mention_strength: Optional[float],
    mention_unit: Optional[str],
    candidate_strength: Optional[float],
    candidate_unit: Optional[str],
    dense_score: float,
) -> ConstraintResult:
    """
    Penalize strength mismatch even when dense score is high.

    Rule: A high dense similarity MUST NOT mask a strength mismatch.
    Dense models sometimes give similar scores to "Metformin 500mg" and
    "Metformin 1000mg" because they share the ingredient. We prevent
    the dense score from compensating for a clear mismatch.
    """
    if mention_strength is None or candidate_strength is None:
        return ConstraintResult(
            passed=True,
            rule="rxnorm_strength_mismatch",
            reason="no strength to compare"
        )

    # Normalize units
    mention_unit_norm = (mention_unit or "").upper().strip()
    cand_unit_norm = (candidate_unit or "").upper().strip()

    # Convert to same unit if needed
    def normalize_strength(val: float, unit: str) -> float:
        u = unit.upper()
        if u in ("G", "GRAM", "GRAMS"):
            return val * 1000
        if u in ("UG", "MCG", "MICROGRAM", "MICROGRAMS"):
            return val / 1000
        return val

    mention_val = normalize_strength(mention_strength, mention_unit_norm)
    cand_val = normalize_strength(candidate_strength, cand_unit_norm)

    if abs(mention_val - cand_val) < 0.01:
        return ConstraintResult(
            passed=True,
            rule="rxnorm_strength_mismatch",
            reason="strength matches"
        )

    # Strength mismatch detected
    mismatch_ratio = abs(mention_val - cand_val) / max(mention_val, cand_val, 0.01)

    # Severity: >50% difference = severe, >20% = moderate, >5% = minor
    if mismatch_ratio > 0.5:
        severity = "severe"
        penalty = 0.5
    elif mismatch_ratio > 0.2:
        severity = "moderate"
        penalty = 0.3
    else:
        severity = "minor"
        penalty = 0.15

    return ConstraintResult(
        passed=False,
        rule="rxnorm_strength_mismatch",
        reason=f"strength mismatch {severity} ({mention_val:.1f} vs {cand_val:.1f}mg), ratio={mismatch_ratio:.2f}",
        penalty=penalty,
    )


def check_rxnorm_dose_form_conflict(
    mention_dose_form: Optional[str],
    candidate_dose_form: Optional[str],
) -> ConstraintResult:
    """
    Penalize dose form conflicts.

    Rule: If mention specifies a dose form (tablet, injection, etc.)
    that contradicts the candidate's dose form, apply penalty.
    """
    if not mention_dose_form or not candidate_dose_form:
        return ConstraintResult(
            passed=True,
            rule="rxnorm_dose_form_conflict",
            reason="no dose form to compare"
        )

    mention_df = mention_dose_form.lower().strip()
    cand_df = candidate_dose_form.lower().strip()

    # Define conflicting pairs
    conflicts = {
        ("tablet", "injection"),
        ("injection", "tablet"),
        ("cream", "tablet"),
        ("tablet", "cream"),
        ("syrup", "tablet"),
        ("tablet", "syrup"),
    }

    if (mention_df, cand_df) in conflicts:
        return ConstraintResult(
            passed=False,
            rule="rxnorm_dose_form_conflict",
            reason=f"dose form conflict: mention='{mention_dose_form}' vs candidate='{candidate_dose_form}'",
            penalty=0.2,
        )

    return ConstraintResult(
        passed=True,
        rule="rxnorm_dose_form_conflict",
        reason="dose forms compatible"
    )


# --- Combined Validator ---


class OntologyValidator:
    """
    Runs all relevant ontology constraints on a candidate.

    Usage:
        validator = OntologyValidator()
        for candidate in candidates:
            result = validator.validate_icd(candidate, mention, entry)
            if not result.passed:
                apply_penalty(candidate, result.penalty)
    """

    def validate_icd(
        self,
        code: str,
        parent_code: Optional[str],
        mention_has_detail: bool,
        mention_normalized: str,
        entry_name: str,
        entry_description: Optional[str],
        include_terms: list[str],
        exclude_terms: list[str],
    ) -> list[ConstraintResult]:
        """
        Validate an ICD-10 candidate against ontology rules.

        Returns list of ConstraintResult (one per rule).
        """
        results = []

        results.append(check_icd_child_context(
            code=code,
            parent_code=parent_code,
            mention_has_detail=mention_has_detail,
            entry_name=entry_name,
            entry_description=entry_description,
        ))

        results.append(check_icd_exclude_terms(
            mention_normalized=mention_normalized,
            exclude_terms=exclude_terms,
        ))

        results.append(check_icd_include_terms(
            mention_normalized=mention_normalized,
            include_terms=include_terms,
            alias_texts=[],
        ))

        return results

    def validate_rxnorm(
        self,
        mention_strength: Optional[float],
        mention_unit: Optional[str],
        mention_dose_form: Optional[str],
        candidate_strength: Optional[float],
        candidate_unit: Optional[str],
        candidate_dose_form: Optional[str],
        dense_score: float,
    ) -> list[ConstraintResult]:
        """
        Validate an RxNorm candidate against ontology rules.

        Returns list of ConstraintResult (one per rule).
        """
        results = []

        results.append(check_rxnorm_strength_mismatch(
            mention_strength=mention_strength,
            mention_unit=mention_unit,
            candidate_strength=candidate_strength,
            candidate_unit=candidate_unit,
            dense_score=dense_score,
        ))

        results.append(check_rxnorm_dose_form_conflict(
            mention_dose_form=mention_dose_form,
            candidate_dose_form=candidate_dose_form,
        ))

        return results

    def total_penalty(self, results: list[ConstraintResult]) -> float:
        """Sum all penalties from constraint results."""
        return sum(r.penalty for r in results)

    def total_boost(self, results: list[ConstraintResult]) -> float:
        """Sum all boosts from constraint results."""
        return sum(r.boost for r in results)
