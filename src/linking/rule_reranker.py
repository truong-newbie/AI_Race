"""
Rule-Based Candidate Reranker for ICD-10 and RxNorm

GIAI ĐOẠN 1: Rule reranker using hand-crafted features.

ICD-10 features:
  - Lexical similarity (mention vs alias/synonym/name)
  - Exact alias match
  - Context attribute match (include/exclude terms)
  - Parent/child consistency
  - Specificity penalty (child code without detail)
  - Specificity boost (include term match)

RxNorm features:
  - Ingredient match
  - Strength exact/close match
  - Unit match
  - Dose form match
  - Brand match
  - Combination coverage

Ontology constraints are applied on top of rule scores.
"""

from dataclasses import dataclass
from typing import Optional
import re

from src.linking.base_reranker import BaseReranker, RerankResult
from src.linking.ontology_constraints import OntologyValidator
from src.linking.icd.schema import ICD10Entry
from src.linking.icd.hybrid_retriever import CandidateResult
from src.linking.icd.preprocess import TextNormalizer
from src.linking.rxnorm.schema import RxNormEntry
from src.linking.rxnorm.parser import DrugMentionParser
from src.linking.rxnorm.normalizer import DrugTextNormalizer


# ─── Weights ──────────────────────────────────────────────────────────────────
# ICD retrieval is already at ~100% R@1, so reranking must be maximally
# conservative. Retrieval score is the dominant signal; rule-based features
# only apply as tie-breakers for near-equal retrieval scores.

ICD_WEIGHTS = {
    "lexical_similarity": 0.02,
    "exact_alias": 0.02,
    "context_include": 0.01,
    "specificity_penalty": -0.01,
    "parent_child_boost": 0.01,
    "retrieval_score": 0.94,    # Near-absolute dominance — preserve retrieval order
}

RX_WEIGHTS = {
    "ingredient": 0.40,
    "strength_exact": 0.20,
    "strength_close": 0.10,
    "unit_match": 0.10,
    "dose_form": 0.10,
    "brand": 0.05,
    "retrieval_score": 0.05,
}


# ─── ICD-10 Rule Reranker ────────────────────────────────────────────────────


@dataclass
class ICDRerankResult(RerankResult):
    """ICD-10 rerank result with ICD-specific fields."""
    is_child_code: bool = False
    constraint_penalty: float = 0.0


class ICDRuleReranker(BaseReranker):
    """
    Rule-based reranker for ICD-10 candidates.

    Features:
    - Lexical similarity: mention text vs alias/synonym/name
    - Exact alias match: binary bonus for exact alias match
    - Context attribute: include/exclude term matching
    - Specificity penalty: child codes without detail
    - Parent/child boost: parent-child relationship consistency
    - Retrieval score: preserve retrieval ranking signal
    """

    def __init__(self, entries: list[ICD10Entry]):
        self.entries = {e.code: e for e in entries}
        self.normalizer = TextNormalizer()
        self.validator = OntologyValidator()

    def name(self) -> str:
        return "icd_rule_reranker"

    def rerank(
        self,
        candidates: list[CandidateResult],
        query: str,
        mention: Optional[str] = None,
        top_k: int = 10,
    ) -> list[ICDRerankResult]:
        if not candidates:
            return []

        self.set_retrieval_batch(candidates)
        text = mention if mention else query
        text_norm = self.normalizer.normalize_for_fuzzy(text)

        results = []
        for idx, c in enumerate(candidates):
            entry = self.entries.get(c.code)
            if entry is None:
                continue

            features, source, is_child, constraint_penalty = self._score_icd(
                c, entry, text, text_norm
            )

            total = sum(
                ICD_WEIGHTS.get(f, 0) * v
                for f, v in features.items()
            )
            # Scale constraint penalty: don't override high-confidence retrieval.
            # If retrieval_score normalized > 0.9, halve the penalty.
            retrieval_conf = features.get("retrieval_score", 0.0)
            penalty_scale = 0.3 if retrieval_conf > 0.9 else 0.8
            total -= abs(constraint_penalty) * penalty_scale
            total = max(0.0, total)

            results.append(ICDRerankResult(
                code=c.code,
                rerank_score=total,
                features=features,
                source=source,
                rank_before=idx + 1,
                is_child_code=is_child,
                constraint_penalty=constraint_penalty,
            ))

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]

    def _score_icd(
        self,
        candidate: CandidateResult,
        entry: ICD10Entry,
        text: str,
        text_norm: str,
    ) -> tuple[dict[str, float], str, bool, float]:
        features: dict[str, float] = {}
        is_child = entry.parent_code is not None and entry.parent_code != entry.code

        # 1. Lexical similarity: best match among all searchable texts
        features["lexical_similarity"] = self._lexical_similarity(text_norm, entry)
        features["retrieval_score"] = self._normalize_retrieval(candidate.score)

        # 2. Exact alias match
        features["exact_alias"] = self._exact_alias_match(text, entry)

        # 3. Context include/exclude
        include_boost, exclude_penalty = self._context_attributes(text_norm, entry)
        features["context_include"] = include_boost
        features["specificity_penalty"] = exclude_penalty

        # 4. Parent/child consistency
        features["parent_child_boost"] = self._parent_child_boost(
            entry, is_child, text
        )

        # Ontology constraints
        constraint_results = self.validator.validate_icd(
            code=entry.code,
            parent_code=entry.parent_code,
            mention_has_detail=self._has_detail(text),
            mention_normalized=text_norm,
            entry_name=entry.name_en or "",
            entry_description=entry.description,
            include_terms=entry.include_terms,
            exclude_terms=entry.exclude_terms,
        )
        constraint_penalty = self.validator.total_penalty(constraint_results)
        constraint_boost = self.validator.total_boost(constraint_results)

        # Primary source
        if features["exact_alias"] > 0:
            source = "exact_alias"
        elif features["lexical_similarity"] > 0.5:
            source = "lexical_similarity"
        elif constraint_boost > 0:
            source = "context_include"
        elif is_child and constraint_penalty > 0:
            source = "specificity_penalty"
        else:
            source = "retrieval_score"

        # Apply constraint boost to features
        features["context_include"] += constraint_boost

        return features, source, is_child, constraint_penalty

    def _lexical_similarity(self, text_norm: str, entry: ICD10Entry) -> float:
        """Best lexical similarity between mention and any searchable text."""
        all_texts = entry.get_all_searchable_texts()
        best = 0.0

        for t in all_texts:
            t_norm = self.normalizer.normalize_for_fuzzy(t)
            score = self._jaccard_words(text_norm, t_norm)
            score = max(score, self._partial_overlap(text_norm, t_norm))
            best = max(best, score)

        # Also check against name_en/name_vi directly
        if entry.name_en:
            norm_en = self.normalizer.normalize_for_fuzzy(entry.name_en)
            best = max(best, self._jaccard_words(text_norm, norm_en))
        if entry.name_vi:
            norm_vi = self.normalizer.normalize_for_fuzzy(entry.name_vi)
            best = max(best, self._jaccard_words(text_norm, norm_vi))

        return best

    def _jaccard_words(self, text1: str, text2: str) -> float:
        """Jaccard similarity between word sets."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        inter = len(words1 & words2)
        union = len(words1 | words2)
        return inter / union if union > 0 else 0.0

    def _partial_overlap(self, text1: str, text2: str) -> float:
        """Fraction of text1 words that appear in text2."""
        words1 = set(text1.lower().split())
        if not words1:
            return 0.0
        matches = sum(1 for w in words1 if w in text2.lower())
        return matches / len(words1)

    def _exact_alias_match(self, text: str, entry: ICD10Entry) -> float:
        """Check for exact alias/synonym match."""
        text_lower = text.lower().strip()
        for alias in entry.aliases:
            if alias.lower().strip() == text_lower:
                return 1.0
        for syn in entry.synonyms:
            if syn.lower().strip() == text_lower:
                return 0.9
        return 0.0

    def _context_attributes(
        self,
        text_norm: str,
        entry: ICD10Entry,
    ) -> tuple[float, float]:
        """Score include/exclude term matches."""
        include_boost = 0.0
        exclude_penalty = 0.0

        for term in entry.include_terms:
            if term.lower() in text_norm:
                include_boost += 0.1

        for term in entry.exclude_terms:
            if term.lower() in text_norm:
                exclude_penalty -= 0.15

        return min(include_boost, 0.25), min(abs(exclude_penalty), 0.30)

    def _parent_child_boost(
        self,
        entry: ICD10Entry,
        is_child: bool,
        text: str,
    ) -> float:
        """Boost parent-child relationship consistency."""
        if not is_child:
            return 0.05  # mild boost for parent codes

        # Check if mention is generic (single word, short)
        if len(text.split()) <= 2 and len(text) < 25:
            # Generic mention — penalize child codes
            return -0.10

        return 0.0

    def _has_detail(self, text: str) -> bool:
        """Check if mention has sufficient detail (drug-specific signals)."""
        detail_signals = [
            "mg", "ml", "g/", "lần", "ngày", "buổi", "liều",
            "uống", "tiêm", "viên", "gói", "tremor", "đau đầu",
            "đau ngực", "khó thở", "sốt", "ho",
        ]
        text_lower = text.lower()
        return any(s in text_lower for s in detail_signals)

    def _normalize_retrieval(self, score: float) -> float:
        """Normalize retrieval score to [0, 1] range using per-batch max."""
        # Must be called with batch context set via set_retrieval_batch()
        if not hasattr(self, '_retrieval_batch_max') or self._retrieval_batch_max == 0:
            return 1.0 if score > 0 else 0.0
        return min(score / self._retrieval_batch_max, 1.0)

    def set_retrieval_batch(self, candidates: list) -> None:
        """Set max retrieval score for this batch (call before reranking)."""
        max_score = max((c.score for c in candidates if hasattr(c, 'score')), default=0.0)
        self._retrieval_batch_max = max_score


# ─── RxNorm Rule Reranker ────────────────────────────────────────────────────


@dataclass
class RxRerankResult(RerankResult):
    """RxNorm rerank result."""
    pass


class RxNormRuleReranker(BaseReranker):
    """
    Rule-based reranker for RxNorm candidates.

    Features:
    - Ingredient match (exact, canonical, brand expansion)
    - Strength exact/close match
    - Unit match
    - Dose form match
    - Brand match
    - Combination coverage
    """

    def __init__(self, entries: list[RxNormEntry]):
        self.entries = {e.rxcui: e for e in entries}
        self.parser = DrugMentionParser()
        self.normalizer = DrugTextNormalizer()
        self.validator = OntologyValidator()

    def name(self) -> str:
        return "rxnorm_rule_reranker"

    def rerank(
        self,
        candidates: list,
        query: str,
        mention: Optional[str] = None,
        top_k: int = 10,
    ) -> list[RxRerankResult]:
        if not candidates:
            return []

        text = mention if mention else query
        parsed = self.parser.parse(text)

        results = []
        for idx, c in enumerate(candidates):
            entry = self.entries.get(c.rxcui)
            if entry is None:
                continue

            features, source = self._score_rxnorm(c, entry, text, parsed)

            total = sum(
                RX_WEIGHTS.get(f, 0) * v
                for f, v in features.items()
            )
            total = max(0.0, total)

            results.append(RxRerankResult(
                code=c.rxcui,
                rerank_score=total,
                features=features,
                source=source,
                rank_before=idx + 1,
            ))

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]

    def _score_rxnorm(
        self,
        candidate,
        entry: RxNormEntry,
        text: str,
        parsed: Optional[DrugMentionParser],
    ) -> tuple[dict[str, float], str]:
        features: dict[str, float] = {}

        features["retrieval_score"] = self._normalize_retrieval(candidate.score)

        if parsed is None:
            return features, "retrieval_score"

        # 1. Ingredient match
        features["ingredient"] = self._ingredient_score(parsed, entry)

        # 2. Strength
        if parsed.has_strength():
            exact, close = self._strength_score(parsed, entry)
            features["strength_exact"] = exact
            features["strength_close"] = close
        else:
            features["strength_exact"] = 0.0
            features["strength_close"] = 0.0

        # 3. Unit
        features["unit_match"] = self._unit_score(parsed, entry)

        # 4. Dose form
        features["dose_form"] = self._dose_form_score(parsed, entry)

        # 5. Brand
        features["brand"] = self._brand_score(parsed, entry)

        # 6. RxNorm ontology constraints
        if parsed.has_strength():
            mention_str_val, mention_str_unit = parsed.main_strength()
        else:
            mention_str_val, mention_str_unit = None, None
        constraint_results = self.validator.validate_rxnorm(
            mention_strength=mention_str_val,
            mention_unit=mention_str_unit,
            mention_dose_form=parsed.dose_form,
            candidate_strength=entry.strength_value,
            candidate_unit=entry.strength_unit,
            candidate_dose_form=entry.dose_form,
            dense_score=candidate.score if hasattr(candidate, 'score') else 0.0,
        )
        constraint_penalty = self.validator.total_penalty(constraint_results)
        if constraint_penalty > 0:
            features["_constraint_penalty"] = constraint_penalty

        # Primary source
        if features["ingredient"] >= 1.0 and features["strength_exact"] > 0:
            source = "ingredient+strength"
        elif features["ingredient"] >= 1.0 and features["strength_close"] > 0:
            source = "ingredient+strength_close"
        elif features["ingredient"] >= 1.0:
            source = "ingredient_only"
        elif features["brand"] > 0:
            source = "brand"
        else:
            source = "retrieval_score"

        return features, source

    def _ingredient_score(self, parsed: DrugMentionParser, entry: RxNormEntry) -> float:
        """Score ingredient match."""
        if parsed.main_ingredient() is None or entry.ingredient is None:
            return 0.0

        parsed_ing = parsed.main_ingredient().lower().strip()
        entry_ing = entry.ingredient.lower().strip()

        # Normalize for comparison
        norm_parsed = self.normalizer.normalize_for_matching(parsed_ing)
        norm_entry = self.normalizer.normalize_for_matching(entry_ing)

        if norm_parsed == norm_entry:
            return 1.0
        if norm_parsed in norm_entry or norm_entry in norm_parsed:
            return 0.8
        return 0.0

    def _strength_score(self, parsed: DrugMentionParser, entry: RxNormEntry) -> tuple[float, float]:
        """Score strength match. Returns (exact, close)."""
        if entry.strength_value is None:
            return 0.0, 0.0

        parsed_str, parsed_unit = parsed.main_strength()
        if parsed_str is None:
            return 0.0, 0.0

        if abs(parsed_str - entry.strength_value) < 0.01:
            return 1.0, 0.0

        # Close match: within 15%
        if parsed_str > 0:
            ratio = abs(parsed_str - entry.strength_value) / parsed_str
            if ratio < 0.15:
                return 0.0, 1.0

        return 0.0, 0.0

    def _unit_score(self, parsed: DrugMentionParser, entry: RxNormEntry) -> float:
        """Score unit match."""
        if not parsed.has_strength() or entry.strength_unit is None:
            return 0.0

        _, unit = parsed.main_strength()
        if unit is None:
            return 0.0

        norm_unit = unit.upper().strip()
        entry_unit = entry.strength_unit.upper().strip()

        if norm_unit == entry_unit:
            return 1.0
        # Accept G=MG*1000 equivalence
        if (norm_unit == "G" and entry_unit == "MG") or (norm_unit == "MG" and entry_unit == "G"):
            return 0.5
        return 0.0

    def _dose_form_score(self, parsed: DrugMentionParser, entry: RxNormEntry) -> float:
        """Score dose form match."""
        if not parsed.dose_form or not entry.dose_form:
            return 0.0

        pf = parsed.dose_form.lower().strip()
        ef = entry.dose_form.lower().strip()

        if pf == ef:
            return 1.0
        # Partial match
        if pf in ef or ef in pf:
            return 0.5
        return 0.0

    def _brand_score(self, parsed: DrugMentionParser, entry: RxNormEntry) -> float:
        """Score brand match."""
        if not parsed.brand or not entry.brand:
            return 0.0

        pb = parsed.brand.lower().strip()
        eb = entry.brand.lower().strip()

        if pb == eb:
            return 1.0
        if pb in eb or eb in pb:
            return 0.7
        return 0.0

    def _normalize_retrieval(self, score: float) -> float:
        """Normalize retrieval score."""
        return min(score / 3.0, 1.0) if score > 0 else 0.0


# ─── Unified Reranker ────────────────────────────────────────────────────────


class UnifiedRuleReranker:
    """
    Unified rule reranker that handles both ICD-10 and RxNorm candidates.

    Automatically detects candidate type based on field names.
    """

    def __init__(
        self,
        icd_entries: Optional[list[ICD10Entry]] = None,
        rx_entries: Optional[list[RxNormEntry]] = None,
    ):
        self.icd_reranker = None
        self.rx_reranker = None
        if icd_entries:
            self.icd_reranker = ICDRuleReranker(icd_entries)
        if rx_entries:
            self.rxnorm_reranker = RxNormRuleReranker(rx_entries)

    def rerank_icd(
        self,
        candidates: list[CandidateResult],
        query: str,
        mention: Optional[str] = None,
        top_k: int = 10,
    ) -> list[ICDRerankResult]:
        if self.icd_reranker is None:
            raise ValueError("ICD reranker not initialized")
        return self.icd_reranker.rerank(candidates, query, mention, top_k)

    def rerank_rxnorm(
        self,
        candidates: list,
        query: str,
        mention: Optional[str] = None,
        top_k: int = 10,
    ) -> list[RxRerankResult]:
        if self.rxnorm_reranker is None:
            raise ValueError("RxNorm reranker not initialized")
        return self.rxnorm_reranker.rerank(candidates, query, mention, top_k)

    def detect_and_rerank(
        self,
        candidates: list,
        query: str,
        mention: Optional[str] = None,
        top_k: int = 10,
    ) -> list[RerankResult]:
        """
        Detect candidate type and route to appropriate reranker.

        Auto-detection: if candidates have 'rxcui' field -> RxNorm,
        if they have 'code' field -> ICD-10.
        """
        if not candidates:
            return []

        # Detect type from first candidate
        first = candidates[0]
        if hasattr(first, "rxcui"):
            return self.rerank_rxnorm(candidates, query, mention, top_k)
        elif hasattr(first, "code"):
            return self.rerank_icd(candidates, query, mention, top_k)
        else:
            raise ValueError(f"Unknown candidate type: {type(first)}")
