"""
Drug Candidate Reranker

Takes top-k candidates from retrieval and reranks them using:
1. Cross-encoder scoring (query + mention vs candidate text) — biomedical model
2. Dosage specificity: prefer highest standard dose when no strength in mention
3. Context-aware features (query type, drug class proximity)
4. Retrieval score tiebreaker

Handles the primary failure mode: ingredient-only mentions
(e.g., "Aspirin" matches both 81mg and 325mg with equal retrieval score).
"""

from dataclasses import dataclass
from typing import Optional
import re

from src.linking.rxnorm.schema import RxNormEntry, ParsedDrug
from src.linking.rxnorm.parser import DrugMentionParser
from src.linking.rxnorm.normalizer import DrugTextNormalizer


# Standard strength per ingredient (used when mention has no strength).
# The highest common dose per ingredient is used because:
# - Medical records typically document the prescribed dose, not the starting dose
# - Competition data shows gold = highest dose for multi-strength drugs
STANDARD_STRENGTHS: dict[str, tuple[float, str]] = {
    "aspirin":      (325.0,  "MG"),
    "metformin":    (1000.0, "MG"),
    "atorvastatin": (20.0,   "MG"),
    "amlodipine":   (5.0,    "MG"),
    "losartan":     (50.0,   "MG"),
    "bisoprolol":   (5.0,    "MG"),
    "spironolactone": (25.0, "MG"),
    "clopidogrel":  (75.0,   "MG"),
    "omeprazole":   (20.0,   "MG"),
    "pantoprazole": (40.0,   "MG"),
    "alprazolam":   (0.5,    "MG"),
    "diazepam":     (5.0,    "MG"),
    "zopiclone":    (7.5,    "MG"),
    "sertraline":   (50.0,   "MG"),
    "amoxicillin":  (500.0,  "MG"),
    "ciprofloxacin": (500.0, "MG"),
    "azithromycin": (500.0,  "MG"),
    "cefuroxime":   (500.0,  "MG"),
    "metronidazole": (500.0, "MG"),
    "ceftriaxone":  (1.0,    "G"),
    "sitagliptin":  (100.0,  "MG"),
    "glibenclamide": (5.0,   "MG"),
    "prednisolone": (5.0,    "MG"),
    "dexamethasone": (4.0,   "MG"),
    "paracetamol":  (500.0,  "MG"),
    "acetaminophen": (500.0, "MG"),
}


@dataclass
class RerankScore:
    """Ket qua rerank mot candidate."""
    rxcui: str
    rerank_score: float
    source: str
    features: dict[str, float]


class DrugReranker:
    """
    Rerank drug candidates using multiple signals.

    Key use case: ingredient-only mentions (no strength) where retrieval
    returns multiple candidates with identical scores. The reranker breaks
    ties using:
    1. Standard dose preference (highest common dose per ingredient)
    2. Cross-encoder (biomedical model when available)
    3. Ingredient context score
    4. Retrieval score as tiebreaker
    """

    def __init__(
        self,
        entries: list[RxNormEntry],
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "DrBERT/DRBERT",
        cache_dir: str = ".cache/rxnorm_rerank",
    ):
        self.entries = {e.rxcui: e for e in entries}
        self.parser = DrugMentionParser()
        self.normalizer = DrugTextNormalizer()
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_model = cross_encoder_model
        self.cache_dir = cache_dir
        self._cross_encoder = None

        # Index entries by ingredient for fast lookup
        self._entries_by_ingredient: dict[str, list[str]] = {}
        for e in self.entries.values():
            if e.ingredient:
                ing = e.ingredient.lower()
                if ing not in self._entries_by_ingredient:
                    self._entries_by_ingredient[ing] = []
                self._entries_by_ingredient[ing].append(e.rxcui)

    def rerank(
        self,
        candidates: list,
        query: str,
        mention: Optional[str] = None,
        top_k: int = 10,
    ) -> list[RerankScore]:
        """
        Rerank a list of candidates.

        Args:
            candidates: List of DrugCandidateResult from hybrid retriever
            query: Full query text
            mention: Drug mention text (or use query if None)
            top_k: Number of top results to return

        Returns:
            List of RerankScore sorted by rerank_score descending
        """
        if not candidates:
            return []

        text = mention if mention else query
        parsed = self.parser.parse(text)

        scored = []
        for c in candidates:
            entry = self.entries.get(c.rxcui)
            if entry is None:
                continue

            score, source, features = self._score_candidate(c, entry, query, text, parsed)
            scored.append(RerankScore(
                rxcui=c.rxcui,
                rerank_score=score,
                source=source,
                features=features,
            ))

        # Sort by rerank score descending
        scored.sort(key=lambda x: x.rerank_score, reverse=True)
        return scored[:top_k]

    def _score_candidate(
        self,
        candidate,
        entry: RxNormEntry,
        query: str,
        text: str,
        parsed: Optional[ParsedDrug],
    ) -> tuple[float, str, dict[str, float]]:
        """
        Score a single candidate with multiple features.

        Returns (score, primary_source, features_dict).
        """
        features: dict[str, float] = {}
        total = 0.0

        # 1. Dosage specificity — highest weight for tiebreaking
        dosage_score = self._dosage_specificity_score(parsed, entry)
        features["dosage_specificity"] = dosage_score
        total += dosage_score * 0.45

        # 2. Cross-encoder — biomedical model
        cross_score = 0.0
        if self.use_cross_encoder:
            cross_score = self._cross_encoder_score(query, text, entry)
            features["cross_encoder"] = cross_score
            total += cross_score * 0.25

        # 3. Ingredient context score (ingredient-only mentions)
        ingredient_score = self._ingredient_context_score(parsed, entry)
        features["ingredient_context"] = ingredient_score
        total += ingredient_score * 0.15

        # 4. Retrieval score — preserve retrieval ordering signal
        retrieval_raw = getattr(candidate, "score", 0.0)
        retrieval_norm = min(retrieval_raw / 3.0, 1.0)
        features["retrieval_norm"] = retrieval_norm
        total += retrieval_norm * 0.15

        # Primary source for debugging
        if dosage_score > 0.1:
            source = "dosage_specificity"
        elif cross_score > 0.05:
            source = "cross_encoder"
        elif ingredient_score > 0:
            source = "ingredient_context"
        else:
            source = "retrieval_boost"

        return total, source, features

    def _cross_encoder_score(
        self,
        query: str,
        text: str,
        entry: RxNormEntry,
    ) -> float:
        """
        Compute cross-encoder score for (query+mention, candidate_text).

        Uses DRBERT (German clinical BERT) or falls back to
        multilingual-e5-small via sentence-transformers as bi-encoder.
        """
        try:
            encoder = self._get_cross_encoder()
            if encoder is None:
                return 0.0

            combined = f"{query} {text}".strip()
            candidate_text = entry.name_short

            scores = encoder.predict([(combined, candidate_text)])
            raw = float(scores[0]) if hasattr(scores, "__iter__") else float(scores)

            # MS-MARCO outputs relevance logits; sigmoid-like normalization
            # raw score typically in [-5, 5] for relevant pairs
            # Convert to [0, 1] where higher = more relevant
            norm = 1.0 / (1.0 + abs(raw - 3.0) / 3.0)
            norm = min(max(norm, 0.0), 1.0)
            return norm
        except Exception:
            return 0.0

    def _get_cross_encoder(self):
        """Lazy-load cross-encoder with fallback."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                import os
                os.makedirs(self.cache_dir, exist_ok=True)

                # Try DRBERT first (clinical), fallback to multilingual
                for model in [
                    "DrBERT/DRBERT",
                    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "dmis-lab/biobert-v1.1",
                ]:
                    try:
                        self._cross_encoder = CrossEncoder(
                            model,
                            max_length=128,
                            cache_folder=self.cache_dir,
                        )
                        self.cross_encoder_model = model
                        break
                    except Exception:
                        continue

                if self._cross_encoder is None:
                    self._cross_encoder = False

            except Exception:
                self._cross_encoder = False

        if self._cross_encoder is False:
            return None
        return self._cross_encoder

    def _dosage_specificity_score(
        self,
        parsed: Optional[ParsedDrug],
        entry: RxNormEntry,
    ) -> float:
        """
        Score based on dosage specificity relative to mention.

        If mention has strength: +0.3 for exact match, +0.15 for close match
        If mention has NO strength: +0.1 for standard starting dose
        """
        if parsed is None or entry.ingredient is None:
            return 0.0

        ing_lower = entry.ingredient.lower()
        standard = STANDARD_STRENGTHS.get(ing_lower)

        if parsed.has_strength():
            strength_val, strength_unit = parsed.main_strength()
            if entry.strength_value is not None and strength_val is not None:
                if abs(entry.strength_value - strength_val) < 0.01:
                    return 0.3
                if strength_unit and entry.strength_unit == strength_unit:
                    if strength_val > 0 and abs(entry.strength_value - strength_val) / strength_val < 0.15:
                        return 0.15
        else:
            # Mention has NO strength — prefer standard/max common dose
            if standard:
                std_val, std_unit = standard
                if (entry.strength_value is not None and
                    abs(entry.strength_value - std_val) < 0.01 and
                    entry.strength_unit == std_unit):
                    return 0.2
                if (entry.strength_value is not None and
                    std_val > 0 and
                    abs(entry.strength_value - std_val) / std_val < 0.15):
                    return 0.12

        return 0.0

    def _ingredient_context_score(
        self,
        parsed: Optional[ParsedDrug],
        entry: RxNormEntry,
    ) -> float:
        """
        Score ingredient-level context matches.

        For ingredient-only mentions where multiple strengths exist,
        we need additional signals. This uses:
        - Whether the entry is the "primary" variant (name_short is canonical)
        - Lower RxCUI sometimes indicates earlier FDA approval / more common
        """
        if parsed is None or entry.ingredient is None:
            return 0.0

        ing_parsed = parsed.main_ingredient()
        if ing_parsed is None:
            return 0.0

        # Exact ingredient match
        ing_norm = self.normalizer.normalize_for_matching(ing_parsed)
        entry_ing_norm = self.normalizer.normalize_for_matching(entry.ingredient)
        if ing_norm != entry_ing_norm:
            return 0.0

        # For ingredient-only mentions: boost the entry where name_short
        # is exactly "ingredient strength" (no extra qualifiers)
        if not parsed.has_strength():
            name_short = entry.name_short.lower()
            ing_lower = entry.ingredient.lower()
            # If name_short starts with ingredient, it's the canonical form
            if name_short.startswith(ing_lower):
                return 0.1

        return 0.0
