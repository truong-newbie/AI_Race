"""
Fuzzy Drug Retriever

RapidFuzz WRatio fallback cho drug retrieval.
"""

from typing import Optional

from rapidfuzz import fuzz

from src.linking.rxnorm.schema import RxNormEntry
from src.linking.rxnorm.normalizer import DrugTextNormalizer


class FuzzyDrugRetriever:
    """
    Fuzzy string matching cho drug retrieval.
    Su dung RapidFuzz WRatio de handle typos va variations.
    """

    def __init__(
        self,
        entries: list[RxNormEntry],
        score_cutoff: int = 50,
        normalizer: Optional[DrugTextNormalizer] = None,
    ):
        self.entries = entries
        self.score_cutoff = score_cutoff
        self.normalizer = normalizer or DrugTextNormalizer()

        # Build searchable texts + codes
        self.texts: list[str] = []
        self.codes: list[str] = []
        self.entry_map: dict[int, RxNormEntry] = {}

        for entry in entries:
            for text in entry.get_all_searchable_texts():
                idx = len(self.texts)
                self.texts.append(text)
                self.codes.append(entry.rxcui)
                self.entry_map[idx] = entry

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Fuzzy retrieve drugs by query.

        Returns list of (rxcui, score) sorted by score descending.
        """
        if not query:
            return []

        norm_query = self.normalizer.normalize_for_matching(query)

        results: list[tuple[str, float, str]] = []
        for i, text in enumerate(self.texts):
            norm_text = self.normalizer.normalize_for_matching(text)
            score = fuzz.WRatio(norm_query, norm_text)
            if score >= self.score_cutoff:
                results.append((self.codes[i], score, text))

        if not results:
            return []

        # Aggregate by rxcui: keep max score
        code_scores: dict[str, float] = {}
        for rxcui, score, _ in results:
            if rxcui not in code_scores or score > code_scores[rxcui]:
                code_scores[rxcui] = score

        ranked = sorted(code_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def retrieve_one(self, query: str) -> Optional[tuple[str, float]]:
        """Retrieve single best match."""
        results = self.retrieve(query, top_k=1)
        if results:
            return results[0]
        return None
