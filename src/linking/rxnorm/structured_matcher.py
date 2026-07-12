"""
Structured Matcher for Drug Retrieval

Priority-based matching:
1. Exact full-name match
2. Exact ingredient + strength + dose form
3. Ingredient + strength
4. Ingredient + dose form
5. Ingredient-only
6. Brand match

With weighted scoring:
- ingredient match: highest weight
- strength exact: high bonus
- strength mismatch: strong penalty
- unit mismatch: strong penalty
- dose form match: bonus
- brand match: bonus
"""

from dataclasses import dataclass
from typing import Optional

from src.linking.rxnorm.schema import RxNormEntry, ParsedDrug
from src.linking.rxnorm.parser import DrugMentionParser
from src.linking.rxnorm.normalizer import DrugTextNormalizer


@dataclass
class MatchScore:
    """Ket qua match mot entry voi mot candidate."""
    rxcui: str
    total_score: float
    ingredient_score: float
    strength_score: float
    dose_form_score: float
    brand_score: float
    source: str
    match_details: str


class StructuredMatcher:
    """
    Structured drug matching với weighted scoring.

    Chi tra ve candidates co ton tai trong RxNorm KB.
    """

    # Score weights
    WEIGHT_INGREDIENT = 0.50
    WEIGHT_STRENGTH_EXACT = 0.25
    WEIGHT_STRENGTH_CLOSE = 0.10
    WEIGHT_UNIT = 0.10
    WEIGHT_DOSE_FORM = 0.05
    WEIGHT_BRAND = 0.05
    WEIGHT_EXACT_FULL = 1.00

    def __init__(
        self,
        entries: list[RxNormEntry],
        parser: Optional[DrugMentionParser] = None,
        normalizer: Optional[DrugTextNormalizer] = None,
    ):
        self.entries = {e.rxcui: e for e in entries}
        self.parser = parser or DrugMentionParser()
        self.normalizer = normalizer or DrugTextNormalizer()
        self._build_indices()

    def _build_indices(self) -> None:
        """Build lookup indices."""
        # name -> rxcui
        self.name_index: dict[str, str] = {}
        # ingredient -> list of rxcuis
        self.ingredient_index: dict[str, list[str]] = {}
        # ingredient + strength -> list of rxcuis
        self.ingredient_strength_index: dict[str, list[str]] = {}
        # brand -> rxcui
        self.brand_index: dict[str, str] = {}
        # normalized name -> rxcui
        self.norm_name_index: dict[str, str] = {}

        for entry in self.entries.values():
            # Index by name
            norm = self.normalizer.normalize_for_matching(entry.name)
            if norm:
                self.norm_name_index[norm] = entry.rxcui

            # Index by ingredient
            if entry.ingredient:
                ing_norm = self.normalizer.normalize_for_matching(entry.ingredient)
                if ing_norm not in self.ingredient_index:
                    self.ingredient_index[ing_norm] = []
                self.ingredient_index[ing_norm].append(entry.rxcui)

                # Index by ingredient + strength
                if entry.strength_value is not None and entry.strength_unit:
                    key = f"{ing_norm}|{entry.strength_value}|{entry.strength_unit}"
                    if key not in self.ingredient_strength_index:
                        self.ingredient_strength_index[key] = []
                    self.ingredient_strength_index[key].append(entry.rxcui)

            # Index by brand
            if entry.brand:
                brand_norm = self.normalizer.normalize_for_matching(entry.brand)
                self.brand_index[brand_norm] = entry.rxcui

    def match(
        self,
        parsed: ParsedDrug,
        top_k: int = 20,
    ) -> list[tuple[str, MatchScore]]:
        """
        Match parsed drug mention against KB.

        Returns list of (rxcui, MatchScore) sorted by total_score descending.
        """
        candidates: dict[str, MatchScore] = {}

        # 1. Exact full-name match
        if parsed.original:
            norm_original = self.normalizer.normalize_for_matching(parsed.original)
            if norm_original in self.norm_name_index:
                rxcui = self.norm_name_index[norm_original]
                candidates[rxcui] = MatchScore(
                    rxcui=rxcui,
                    total_score=self.WEIGHT_EXACT_FULL,
                    ingredient_score=self.WEIGHT_EXACT_FULL,
                    strength_score=0.0,
                    dose_form_score=0.0,
                    brand_score=0.0,
                    source="exact_full",
                    match_details="exact name match",
                )

        # 2. Exact ingredient + strength + dose form
        ing = parsed.main_ingredient()
        if ing and parsed.has_strength():
            strength_val, strength_unit = parsed.main_strength()
            if strength_val is not None and strength_unit:
                ing_norm = self.normalizer.normalize_for_matching(ing)
                key = f"{ing_norm}|{strength_val}|{strength_unit}"
                if key in self.ingredient_strength_index:
                    for rxcui in self.ingredient_strength_index[key]:
                        entry = self.entries[rxcui]
                        score = self._score_entry(parsed, entry)
                        if rxcui not in candidates or score.total_score > candidates[rxcui].total_score:
                            candidates[rxcui] = score

        # 3. Ingredient + strength (without dose form)
        if ing and parsed.has_strength() and not candidates:
            strength_val, strength_unit = parsed.main_strength()
            if strength_val is not None:
                ing_norm = self.normalizer.normalize_for_matching(ing)
                for rxcui in self.ingredient_index.get(ing_norm, []):
                    entry = self.entries[rxcui]
                    if (entry.strength_value is not None and
                            abs(entry.strength_value - strength_val) < 0.01 and
                            (strength_unit is None or entry.strength_unit == strength_unit)):
                        score = self._score_entry(parsed, entry)
                        if rxcui not in candidates or score.total_score > candidates[rxcui].total_score:
                            candidates[rxcui] = score

        # 4. Ingredient + dose form
        if ing and parsed.dose_form:
            ing_norm = self.normalizer.normalize_for_matching(ing)
            for rxcui in self.ingredient_index.get(ing_norm, []):
                entry = self.entries[rxcui]
                if entry.dose_form == parsed.dose_form:
                    score = self._score_entry(parsed, entry)
                    if rxcui not in candidates or score.total_score > candidates[rxcui].total_score:
                        candidates[rxcui] = score

        # 5. Ingredient-only
        if ing and not candidates:
            ing_norm = self.normalizer.normalize_for_matching(ing)
            for rxcui in self.ingredient_index.get(ing_norm, []):
                entry = self.entries[rxcui]
                score = self._score_entry(parsed, entry)
                if rxcui not in candidates or score.total_score > candidates[rxcui].total_score:
                    candidates[rxcui] = score

        # 6. Brand match
        if parsed.brand:
            brand_norm = self.normalizer.normalize_for_matching(parsed.brand)
            if brand_norm in self.brand_index:
                rxcui = self.brand_index[brand_norm]
                entry = self.entries[rxcui]
                score = self._score_entry(parsed, entry)
                if rxcui not in candidates or score.total_score > candidates[rxcui].total_score:
                    candidates[rxcui] = score

        # Sort by total score descending
        ranked = sorted(candidates.items(), key=lambda x: x[1].total_score, reverse=True)
        return ranked[:top_k]

    def _score_entry(
        self,
        parsed: ParsedDrug,
        entry: RxNormEntry,
    ) -> MatchScore:
        """Calculate weighted score for a candidate entry."""
        ing_score = 0.0
        strength_score = 0.0
        unit_score = 0.0
        dose_form_score = 0.0
        brand_score = 0.0
        details_parts = []
        source = "structured"

        # Ingredient match (highest weight)
        ing = parsed.main_ingredient()
        if ing:
            ing_norm = self.normalizer.normalize_for_matching(ing)
            entry_ing_norm = self.normalizer.normalize_for_matching(entry.ingredient or "")
            if ing_norm == entry_ing_norm:
                ing_score = self.WEIGHT_INGREDIENT
                details_parts.append("ingredient_match")
            elif entry_ing_norm and (ing_norm in entry_ing_norm or entry_ing_norm in ing_norm):
                ing_score = self.WEIGHT_INGREDIENT * 0.7
                details_parts.append("ingredient_partial")

        # Strength match
        if parsed.has_strength():
            strength_val, strength_unit = parsed.main_strength()
            if entry.strength_value is not None and strength_val is not None:
                # Exact strength match
                if abs(entry.strength_value - strength_val) < 0.01:
                    strength_score = self.WEIGHT_STRENGTH_EXACT
                    details_parts.append("strength_exact")
                # Close strength match (< 10% difference)
                elif strength_val > 0 and abs(entry.strength_value - strength_val) / strength_val < 0.1:
                    strength_score = self.WEIGHT_STRENGTH_CLOSE
                    details_parts.append("strength_close")
                else:
                    # Strength mismatch: penalize
                    strength_score = -0.3
                    details_parts.append("strength_mismatch")
            elif entry.strength_value is None and strength_val is not None:
                # Entry has no strength, penalize if mention has strength
                strength_score = -0.2
                details_parts.append("strength_missing")

            # Unit match
            if strength_unit and entry.strength_unit:
                if strength_unit == entry.strength_unit:
                    unit_score = self.WEIGHT_UNIT
                    details_parts.append("unit_match")
                else:
                    unit_score = -0.1
                    details_parts.append("unit_mismatch")
            elif strength_unit and not entry.strength_unit:
                unit_score = -0.05

        # Dose form match
        if parsed.dose_form and entry.dose_form:
            if parsed.dose_form == entry.dose_form:
                dose_form_score = self.WEIGHT_DOSE_FORM
                details_parts.append("dose_form_match")

        # Brand match
        if parsed.brand and entry.brand:
            brand_norm = self.normalizer.normalize_for_matching(parsed.brand)
            entry_brand_norm = self.normalizer.normalize_for_matching(entry.brand)
            if brand_norm == entry_brand_norm:
                brand_score = self.WEIGHT_BRAND
                details_parts.append("brand_match")

        total = ing_score + strength_score + unit_score + dose_form_score + brand_score
        total = max(0.0, min(1.0, total))

        return MatchScore(
            rxcui=entry.rxcui,
            total_score=total,
            ingredient_score=ing_score,
            strength_score=strength_score,
            dose_form_score=dose_form_score,
            brand_score=brand_score,
            source=source,
            match_details=" + ".join(details_parts) if details_parts else "partial_match",
        )

    def lookup_exact(self, text: str) -> list[str]:
        """Direct lookup by exact name."""
        norm = self.normalizer.normalize_for_matching(text)
        if norm in self.norm_name_index:
            return [self.norm_name_index[norm]]
        return []
