"""
Base reranker abstract class for ICD-10 and RxNorm candidate reranking.

All rerankers (rule-based, cross-encoder, hybrid) inherit from this.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class RerankResult:
    """
    Reranked candidate result with score breakdown.

    Attributes:
        code: ICD-10 code or RxNorm RxCUI
        rerank_score: Final combined score after reranking
        features: Per-feature scores (lexical, dense, alias, etc.)
        source: Primary signal that determined the ranking
        rank_before: Original rank in retrieval (None if not available)
    """
    code: str
    rerank_score: float
    features: dict[str, float] = field(default_factory=dict)
    source: str = ""
    rank_before: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "rerank_score": round(self.rerank_score, 4),
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "source": self.source,
            "rank_before": self.rank_before,
        }


class BaseReranker(ABC):
    """
    Abstract base for all candidate rerankers.

    Subclasses must implement:
        rerank(candidates, query, mention, top_k) -> list[RerankResult]
        name() -> str
    """

    @abstractmethod
    def rerank(
        self,
        candidates: list[Any],
        query: str,
        mention: Optional[str] = None,
        top_k: int = 10,
    ) -> list[RerankResult]:
        """
        Rerank a list of retrieval candidates.

        Args:
            candidates: List of candidate objects from retrieval.
                For ICD: CandidateResult(code, score, sources, detail)
                For RxNorm: DrugCandidateResult(rxcui, score, sources)
            query: Full query/sentence text
            mention: Entity mention text (or None to use query)
            top_k: Number of top results to return

        Returns:
            List of RerankResult sorted by rerank_score descending
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Short name of this reranker."""
        pass
