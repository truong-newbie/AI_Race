"""
BM25 Retriever for ICD-10 Candidate Retrieval

Okapi BM25 ranking over all searchable texts per entry.
"""

from typing import Optional
from rank_bm25 import BM25Okapi
from src.linking.icd.schema import ICD10Entry
from src.linking.icd.preprocess import TextNormalizer


class BM25Retriever:
    """BM25-based retrieval over ICD-10 entries."""

    def __init__(
        self,
        entries: Optional[list[ICD10Entry]] = None,
        normalizer: Optional[TextNormalizer] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.normalizer = normalizer or TextNormalizer()
        self.k1 = k1
        self.b = b
        self.entries: list[ICD10Entry] = []
        self.corpus: list[list[str]] = []  # tokenized texts per entry
        self.entry_codes: list[str] = []
        self.bm25: Optional[BM25Okapi] = None

        if entries:
            self.build(entries)

    def build(self, entries: list[ICD10Entry]) -> None:
        """Tokenize all searchable texts and build BM25 index."""
        self.entries = entries
        self.corpus = []
        self.entry_codes = []

        for entry in entries:
            texts = entry.get_all_searchable_texts()
            # Tokenize by whitespace
            tokenized = []
            for text in texts:
                norm = self.normalizer.normalize(text)
                if norm:
                    tokenized.extend(norm.split())
            self.corpus.append(tokenized)
            self.entry_codes.append(entry.code)

        self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Retrieve top-k ICD codes by BM25 score.

        Returns list of (code, score).
        """
        if self.bm25 is None:
            return []

        norm_query = self.normalizer.normalize(query)
        tokens = norm_query.split() if norm_query else []

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored = list(zip(self.entry_codes, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_one(self, query: str) -> Optional[tuple[str, float]]:
        """Retrieve top-1 match."""
        results = self.retrieve(query, top_k=1)
        if results:
            return results[0]
        return None
