"""
Hybrid Retriever — Combines all 6 retrieval sources

Retrieval sources:
  1. Exact match (alias index)
  2. Normalized exact match
  3. Alias match
  4. RapidFuzz fuzzy
  5. BM25
  6. Dense embedding

Merge: weighted normalized score OR reciprocal rank fusion (RRF).

Configurable weights and top-k.
"""

from dataclasses import dataclass, field
from typing import Optional
from src.linking.icd.schema import ICD10Entry, get_knowledge_base
from src.linking.icd.preprocess import TextNormalizer
from src.linking.icd.alias_index import AliasIndex
from src.linking.icd.fuzzy_retriever import FuzzyRetriever
from src.linking.icd.bm25_retriever import BM25Retriever
from src.linking.icd.dense_retriever import DenseRetriever


@dataclass
class RetrievalSource:
    """A single retrieval source result."""
    code: str
    score: float
    source: str


@dataclass
class CandidateResult:
    """Merged candidate with combined score and source breakdown."""
    code: str
    score: float
    sources: list[str] = field(default_factory=list)
    detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "score": round(self.score, 4),
            "sources": self.sources,
        }


@dataclass
class MergeConfig:
    """Configuration for result merging."""
    method: str = "rrf"  # "rrf" or "weighted"
    # RRF
    rrf_k: int = 60
    # Weighted
    weights: dict[str, float] = field(default_factory=lambda: {
        "exact": 1.0,
        "normalized": 0.95,
        "alias": 0.9,
        "fuzzy": 0.8,
        "bm25": 0.7,
        "dense": 0.75,
    })
    # Normalize score range
    fuzzy_max: float = 100.0
    bm25_max: float = 20.0
    dense_max: float = 1.0

    def get_weight(self, source: str) -> float:
        return self.weights.get(source, 0.5)


class HybridRetriever:
    """
    Hybrid ICD-10 candidate retriever.

    Combines 6 sources via RRF or weighted normalized scores.
    """

    def __init__(
        self,
        entries: Optional[list[ICD10Entry]] = None,
        merge_config: Optional[MergeConfig] = None,
        normalizer: Optional[TextNormalizer] = None,
        dense_model: str = "intfloat/multilingual-e5-small",
        fuzzy_score_cutoff: int = 50,
        top_k: int = 20,
    ):
        self.merge_config = merge_config or MergeConfig()
        self.normalizer = normalizer or TextNormalizer()
        self.fuzzy_score_cutoff = fuzzy_score_cutoff
        self.top_k = top_k

        self.alias_index = AliasIndex(self.normalizer)
        self.fuzzy = FuzzyRetriever(normalizer=self.normalizer, score_cutoff=fuzzy_score_cutoff)
        self.bm25 = BM25Retriever(normalizer=self.normalizer)
        self.dense = DenseRetriever(model_name=dense_model)

        self._built = False
        if entries:
            self.build(entries)

    def build(self, entries: list[ICD10Entry]) -> None:
        """Build all sub-indices from entries."""
        self.alias_index.build(entries)
        self.fuzzy.build(entries)
        self.bm25.build(entries)
        self.dense.build(entries)
        self._built = True

    def _normalize_score(self, score: float, source: str) -> float:
        """Normalize score to [0, 1] range."""
        cfg = self.merge_config
        if source == "fuzzy":
            return min(score / cfg.fuzzy_max, 1.0)
        elif source == "bm25":
            return min(score / cfg.bm25_max, 1.0) if cfg.bm25_max > 0 else 0.0
        elif source == "dense":
            return min(score / cfg.dense_max, 1.0) if cfg.dense_max > 0 else 0.0
        elif source in ("exact", "normalized", "alias"):
            return score  # already 0/1 or boost factor
        return score

    def _get_all_sources(self, query: str, mention: str) -> dict[str, list[tuple[str, float]]]:
        """Run all 6 retrieval sources."""
        text = mention  # use mention as primary query
        if not text.strip():
            text = query

        results: dict[str, list[tuple[str, float]]] = {}

        # 1. Exact match (original alias index)
        exact_codes = self.alias_index.lookup_exact(text)
        results["exact"] = [(c, 1.0) for c in exact_codes]

        # 2. Normalized exact
        norm_codes = self.alias_index.lookup_normalized(text)
        results["normalized"] = [(c, 1.0) for c in norm_codes if c not in exact_codes]

        # 3. Alias match
        alias_codes = self.alias_index.lookup_normalized(self.normalizer.normalize_for_alias(text))
        results["alias"] = [(c, 0.9) for c in alias_codes
                            if c not in exact_codes and c not in norm_codes]

        # 4. Fuzzy (RapidFuzz)
        results["fuzzy"] = self.fuzzy.retrieve(text, top_k=30)

        # 5. BM25
        results["bm25"] = self.bm25.retrieve(text, top_k=30)

        # 6. Dense embedding
        results["dense"] = self.dense.retrieve(text, top_k=30)

        return results

    def _merge_rrf(self, source_results: dict[str, list[tuple[str, float]]], top_k: int) -> list[CandidateResult]:
        """Merge using Reciprocal Rank Fusion."""
        k = self.merge_config.rrf_k
        rrf_scores: dict[str, float] = {}

        for source, results in source_results.items():
            weight = self.merge_config.get_weight(source)
            for rank, (code, raw_score) in enumerate(results, start=1):
                norm_score = self._normalize_score(raw_score, source)
                rrf = (1.0 / (k + rank)) * weight * (0.5 + 0.5 * norm_score)
                rrf_scores[code] = rrf_scores.get(code, 0.0) + rrf

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build source breakdown
        code_sources: dict[str, list[str]] = {code: [] for code, _ in ranked}
        for source, results in source_results.items():
            seen: set[str] = set()
            for code, _ in results:
                if code in code_sources and code not in seen:
                    code_sources[code].append(source)
                    seen.add(code)

        return [
            CandidateResult(
                code=code,
                score=score,
                sources=code_sources.get(code, []),
                detail={},
            )
            for code, score in ranked
        ]

    def _merge_weighted(self, source_results: dict[str, list[tuple[str, float]]], top_k: int) -> list[CandidateResult]:
        """Merge using weighted normalized score."""
        weighted_scores: dict[str, tuple[float, dict[str, float]]] = {}

        for source, results in source_results.items():
            weight = self.merge_config.get_weight(source)
            for code, raw_score in results:
                norm = self._normalize_score(raw_score, source)
                contribution = weight * norm
                if code not in weighted_scores:
                    weighted_scores[code] = (0.0, {})
                ws, detail = weighted_scores[code]
                ws += contribution
                detail[source] = norm
                weighted_scores[code] = (ws, detail)

        ranked = sorted(weighted_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

        return [
            CandidateResult(
                code=code,
                score=ws,
                sources=[s for s, v in detail.items() if v > 0],
                detail=detail,
            )
            for code, (ws, detail) in ranked
        ]

    def retrieve(
        self,
        query: str,
        mention: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[CandidateResult]:
        """
        Retrieve top-k ICD-10 candidates.

        Args:
            query: Full sentence or context containing the mention.
            mention: The entity mention text.
            top_k: Override default top_k.

        Returns:
            List of CandidateResult sorted by merged score.
        """
        if not self._built:
            raise RuntimeError("Call build() before retrieve().")

        k = top_k or self.top_k
        source_results = self._get_all_sources(query, mention or query)

        if self.merge_config.method == "rrf":
            return self._merge_rrf(source_results, k)
        else:
            return self._merge_weighted(source_results, k)

    def retrieve_codes(
        self,
        query: str,
        mention: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[str]:
        """Retrieve just the code list (convenience)."""
        return [c.code for c in self.retrieve(query, mention, top_k)]
