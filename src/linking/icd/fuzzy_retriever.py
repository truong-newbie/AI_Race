"""
Fuzzy Retriever using RapidFuzz

Fast approximate string matching for ICD-10 candidates.
"""

from typing import Optional
from rapidfuzz import fuzz, process
from src.linking.icd.schema import ICD10Entry
from src.linking.icd.preprocess import TextNormalizer


class FuzzyRetriever:
    """
    RapidFuzz-based fuzzy retrieval.

    Compares query against all searchable texts of each ICD-10 entry
    and returns top-k matches.
    """

    def __init__(
        self,
        entries: Optional[list[ICD10Entry]] = None,
        normalizer: Optional[TextNormalizer] = None,
        score_cutoff: int = 50,
    ):
        self.normalizer = normalizer or TextNormalizer()
        self.score_cutoff = score_cutoff
        self.entries: list[ICD10Entry] = []
        self._corpus: list[tuple[str, str]] = []  # (normalized_text, code)

        if entries:
            self.build(entries)

    def build(self, entries: list[ICD10Entry]) -> None:
        """Build corpus from entries."""
        self.entries = entries
        self._corpus = []

        for entry in entries:
            for text in entry.get_all_searchable_texts():
                if text:
                    norm = self.normalizer.normalize_for_fuzzy(text)
                    if norm:
                        self._corpus.append((norm, entry.code))

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Retrieve top-k ICD codes by fuzzy matching.

        Returns list of (code, score).
        """
        norm_query = self.normalizer.normalize_for_fuzzy(query)
        if not norm_query or not self._corpus:
            return []

        texts = [item[0] for item in self._corpus]
        codes = [item[1] for item in self._corpus]

        results = process.extract(
            norm_query,
            texts,
            scorer=fuzz.WRatio,
            limit=top_k * 3,
            score_cutoff=self.score_cutoff,
        )

        # Aggregate by code: max score
        code_scores: dict[str, float] = {}
        for matched_text, score, _ in results:
            idx = texts.index(matched_text)
            code = codes[idx]
            if code not in code_scores or score > code_scores[code]:
                code_scores[code] = score

        ranked = sorted(code_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    def retrieve_one(self, query: str) -> Optional[tuple[str, float]]:
        """Retrieve top-1 match."""
        results = self.retrieve(query, top_k=1)
        if results:
            return (results[0][0], results[0][1])
        return None
